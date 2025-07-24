import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import (
    JointGroupCommand,
    JointSingleCommand,
    JointTrajectoryCommand
)

from interbotix_xs_msgs.srv import (
    OperatingModes,
)

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
import time
import math
import socket
import numpy as np
from scipy.linalg import block_diag
import json
import atexit
from scipy.io import loadmat
import scipy

import scipy.io

from Gravity_Compensation_Function import calculate_gravity
import inverse_kinematics
import forward_kinematics


from inverse_kinematics import inverse_kinematics
from forward_kinematics import forward_kinematics
import torch
import torch.nn as nn

import time
# ---------------------------
# Neural Network Definition
# ---------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(4, 600)
        self.layer2 = nn.Linear(600, 300)
        self.layer3 = nn.Linear(300, 150)
        self.layer4 = nn.Linear(150, 75)
        self.layer5 = nn.Linear(75, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))

        return self.output_layer(x)
    


# ---------------------------
# Load the Trained Model
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = NeuralNetwork().to(device)

# Load the  model weights
best_model_path = 'trained_network_manipulator.pth'  
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set to evaluation mode




class ViperX300sController(Node):
    def __init__(self):
        super().__init__('ViperX_300s_Controller')

        # Call the service to set the operating mode
        self.call_service()

        # Create publisher for joint commands
        self.publisher = self.create_publisher(JointGroupCommand, '/vx300s/commands/joint_group', 10)

        # Subscribe to joint states topic
        self.subscription = self.create_subscription(JointState, '/vx300s/joint_states', self.listener_callback, 10)

        # Store the most recent message without processing immediately
        self.recent_joint_state = None

        atexit.register(self.save_data)
        self.joint_data = []



        # Timer to process joint states at a slower rate (166Hz in this case)
        self.timer = self.create_timer(0.006, self.process_joint_state)

        # time for updating the applied reference using ERG
        self.dt_v = 0.2
        self.timer_ERG = self.create_timer(self.dt_v, self.ERG)  


        # Initialize variables for PD control and joint currents
        self.prev_error = [0] * 9
        self.joint_currents = [0] * 9
        self.PD_command = [0] * 9

        # Proportional (Kp) and derivative (Kd) gains for PD control, tuned for each joint


        self.kp = {
            'waist': 4500,
            'shoulder': 5000,
            'elbow': 4700,
            'forearm_roll': 2200,
            'wrist_angle': 1500,
            'wrist_rotate':1000,
        }

        self.kd = {
            'waist': 200,
            'shoulder': 350,
            'elbow': 200,
            'forearm_roll': 10,
            'wrist_angle': 50,
            'wrist_rotate': 25,
        }         







        # Limits for joint currents
        self.u_min = [-3200.0] * 9
        self.u_max = [3200.0] * 9

 
        # fix joints 4 and 6 to zero
        self.th4_d = 0.0
        self.th6_d = 0.0
        



        self.previous_time = time.time()
        #  ERG Parameters:
        self.inc = 0
        
        self.KpMatrix = np.diag([self.kp['waist'], self.kp['shoulder'], self.kp['elbow'], self.kp['forearm_roll'], self.kp['wrist_angle'], self.kp['wrist_rotate']])


        px=600
        py=600
        pz=426.75


        self.r = np.array([px, py, pz]).T # the deired reference
        self.eta1 = 0.01 #smoothing factor
        self.zeta = math.radians(0.8) # this is xi, # The distance from the constraint from which the repulsive term's effect begins
        self.delta = math.radians(0.5) # The distance from the constraint from which the repulsive term's effect is maximum
        self.mu = 1 # lipschitz (sensitivity of the equilibrium point to the change in applied reference)
        self.eta2 = 0.01 #smoothing factor for dynamic kappa
        self.kappa = []  # initialize kappa as an empty list
        self.kappa.append(0.0)

        #  the end effector goes first to this initial location and after 5 seconds it should start tracking the ERG generated applied reference
        self.px_i = -380
        self.py_i = -250
        self.pz_i = 150

        self.Running_time = 0 
        
        # defining the wall constraints f1 & f2
        self.b = 0.0
        self.a = 0.0
        self.k1 = -1.133160000000000e+03
        self.k2 = 1.132840000000000e+03
        self.Beta = 3.115139644589057e-06
        self.Tolerance = 59






        # control command parameters
        beta1 = 0.0
        beta2 = 10
        beta3 = 10
        beta4 = 10
        beta5 = 10
        beta6 = 5



        self.beta = np.array([
            [beta1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, beta2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, beta3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, beta4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, beta5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, beta6]
        ])

        # Friction coefficients used in the control command law
        self.coefficients = np.array([

            1.127192704159674e+01,
            -7.592139176521678e+00,
            -1.214059817341892e+00,
            -5.802166487214266e+00,
            5.326906738275686e+00,
            -1.361027413566556e+01,
            6.448609461690413e+01,
            1.000018422040135e+01,
            7.726490421282266e+00,
            2.811937241823886e+00,
            -2.028501337197468e+00,
            1.080859203473093e+00,
            -4.620877101805478e+00,
            1.521413209726164e+00,
            1.436069115753074e+01,
            4.969966618017289e+00,
            5.708555923394439e+00,
            -1.356794235898829e+00,
            4.796481101274910e+00,
            -3.755739111306564e+00,
            5.274663381701949e+00,
            2.041183206656774e-01,
            3.434816194586145e+00,
            -8.818793425095958e-01,
            -6.693261719670063e-01,
            -7.172163600458437e-01,
            -3.374360711515359e-01,
            -2.327540766551572e+00,
            5.892564113829182e+00,
            -2.656138277067310e+00,
            4.166221408188511e+00,
            -4.412954965588797e+00,
            -4.554881504846105e+00,
            -8.160293608153412e-01,
            3.910889217987887e+00,
            -2.078211662299954e+00,
            7.259100472428752e+01,
            6.635812473376058e+01,
            2.135646068362948e+01,
            1.644223863282695e+02,
            1.248187739562997e-03,
            1.350426695412986e+02,
            6.753911572188992e-04,
            1.176898085971351e+01,
            8.680457573418034e-03,
            1.950457462423874e+01,
            1.556159281879819e-02,
            9.964842235055412e-03,
            -1.321403959066689e+01,
            -2.943904537748044e+01,
            -2.505850231773448e+02,
            -1.178494286011038e+01,
            -1.632120872955739e+01,
            -1.700949877698571e+01])




    # Calls the '/vx300s/set_operating_modes' service to set the operating mode of the robot arm to 'current'.
    def call_service(self):
        client = self.create_client(OperatingModes, '/vx300s/set_operating_modes')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        request = OperatingModes.Request()
        request.cmd_type='group'
        request.name = 'arm'
        request.mode = 'current'
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Service call successful')
        else:
            self.get_logger().error('Service call failed')







    def listener_callback(self, msg):
        # Store the incoming joint state message for processing later
        self.recent_joint_state = msg

    def process_joint_state(self):
        if self.recent_joint_state is None:
            return  # No data to process yet

        joint_names = self.recent_joint_state.name
        joint_positions = self.recent_joint_state.position
        joint_velocities = self.recent_joint_state.velocity
        joint_efforts = self.recent_joint_state.effort




        # self.get_logger().info(f'joint_positions: {joint_positions}')
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time  # update previous time


        self.Running_time = self.Running_time + dt
        #  the end effector goes first to this initial location and after 5 seconds it should start tracking the ERG generated applied reference
        if self.Running_time >= 5:

 



            # the robot measures q, the reference command should be q
            self.desired_positions = {
                'waist': self.th_v[0],
                'shoulder': self.th_v[1] + math.radians(78.69),
                'elbow': self.th_v[2] + math.radians(11.31),
                'forearm_roll': self.th4_d,
                'wrist_angle': self.th_v[4],
                'wrist_rotate': self.th6_d,
            }


        else:
            # initial positions, the robot measures q, the reference command should be q
            q1_i,q2_i,q3_i,q4_i,q5_i,q6_i = inverse_kinematics(self.px_i, self.py_i, self.pz_i)
            # self.get_logger().info(f'q_i : {q1_i,q2_i,q3_i,q4_i,q5_i,q6_i}')

            self.desired_positions = {
                'waist': math.radians(q1_i),
                'shoulder': math.radians(q2_i),
                'elbow': math.radians(q3_i),
                'forearm_roll': math.radians(q4_i),
                'wrist_angle': math.radians(q5_i),
                'wrist_rotate': math.radians(q6_i),
            }







        # PD control logic for joints
        for i, joint_name in enumerate(joint_names[:6]):
            if joint_name in self.desired_positions:
                desired_position = self.desired_positions[joint_name]
                error = desired_position - joint_positions[i]
                p_error = error
                d_error = (error - self.prev_error[i]) / dt

                u = self.kp[joint_name] * p_error + self.kd[joint_name] * d_error



                self.PD_command[i] = u
                self.prev_error[i] = error

        
        q1, q2, q3, q4, q5, q6 = joint_positions[:6]  
        G = calculate_gravity(q1, q2, q3, q4, q5, q6) # get the gravity compensation torques





        q1dot, q2dot, q3dot, q4dot, q5dot, q6dot = joint_velocities[:6]

        qdot = np.array([q1dot, q2dot, q3dot, q4dot, q5dot, q6dot]).T
        
        if np.linalg.norm(qdot) <= 1e-1:

            tau_0 = 0.0 * qdot
            
        else:

            qdot_normalized = qdot / np.linalg.norm(qdot)
            tau_0 = -self.beta @ qdot_normalized





        # compute the friction torques and the torque offsets.

        F1 = self.coefficients[36] * q1dot + self.coefficients[48]
        F2 = self.coefficients[38] * q2dot +  self.coefficients[49]
        F3 = self.coefficients[40] * q3dot +  self.coefficients[50]
        F4 = self.coefficients[42] * q4dot +  self.coefficients[51]
        F5 = self.coefficients[44] * q5dot +  self.coefficients[52]
        F6 = self.coefficients[46] * q6dot +  self.coefficients[53]

        # compute the applied torques to the joints    

        self.joint_currents[0] = max(min(self.PD_command[0] + float(G[0]) + F1 + tau_0[0], 3200.0), -3200.0)
        self.joint_currents[1] = max(min(self.PD_command[1] + float(G[1]) + F2 + tau_0[1], 3200.0), -3200.0)
        self.joint_currents[2] = max(min(self.PD_command[2] + float(G[2]) + F3 + tau_0[2], 3200.0), -3200.0)
        self.joint_currents[3] = max(min(self.PD_command[3] + float(G[3]) + F4 + tau_0[3], 3200.0), -3200.0)
        self.joint_currents[4] = max(min(self.PD_command[4] + float(G[4]) + F5 + tau_0[4], 3200.0), -3200.0)
        self.joint_currents[5] = max(min(self.PD_command[5] + float(G[5]) + F6 + tau_0[5], 3200.0), -3200.0)












        # Publish joint currents as commands
        jointcommand = JointGroupCommand()
        jointcommand.name = 'arm'

        jointcommand.cmd = [self.joint_currents[0], self.joint_currents[1], self.joint_currents[2], self.joint_currents[3], self.joint_currents[4], self.joint_currents[5]]
        self.publisher.publish(jointcommand)



        # collect data for saving later on

        joint_data_entry = {
            'timestamp': current_time,
            'name': joint_names,
            'position': list(joint_positions),
            'velocity': list(joint_velocities),
            'effort': list(joint_efforts),
            'desired_positions': list(self.desired_positions.values()),
            'Kappa': self.kappa[-1]
        }

        self.joint_data.append(joint_data_entry)




    def ERG(self):
        if self.recent_joint_state is None:
            return  # No data to process yet

        joint_names = self.recent_joint_state.name
        joint_positions = self.recent_joint_state.position
        joint_velocities = self.recent_joint_state.velocity










        #  the input values
        q1 = joint_positions[0]
        q2 = joint_positions[1]
        q3 = joint_positions[2]
        q4 = joint_positions[3]
        q5 = joint_positions[4]
        q6 = joint_positions[5]
        
        # send data to MATLAB on the windows side to compute the mass matrix
        input_values = [q1, q2, q3, q4, q5, q6] # in MATLAB they are converted into theta because the model develped is M(theta)...etc
        input_str = ','.join(map(str, input_values))


        # Set up a TCP/IP client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('172.28.160.1', 12345))  #  '172.28.160.1' is the IP address of  Windows machine


        th1 = q1
        th2 = q2 - math.radians(78.69)
        th3 = q3 - math.radians(11.31)
        th4 = q4
        th5 = q5
        th6 = q6

        th1_dot = joint_velocities[0]
        th2_dot = joint_velocities[1]
        th3_dot = joint_velocities[2]
        th4_dot = joint_velocities[3]
        th5_dot = joint_velocities[4]
        th6_dot = joint_velocities[5] 


        Theta = np.array([th1, th2, th3, th4, th5, th6])
        Theta_dot = np.array([th1_dot, th2_dot, th3_dot, th4_dot, th5_dot, th6_dot])  

        self.inc =self.inc+1
        self.get_logger().info(f'inc : {self.inc}')


        if self.inc == 1:


            th_v0 = np.array([th1, th2, th3, th5]).T


            self.th_v = []
            self.v = []


            self.kappa = []


            X_v = np.array([th_v0[0], th_v0[1], th_v0[2], self.th4_d, th_v0[3], self.th6_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T  #the equilibrium state
            X = np.array([Theta[0], Theta[1], Theta[2], Theta[3], Theta[4], Theta[5], Theta_dot[0], Theta_dot[1], Theta_dot[2], Theta_dot[3], Theta_dot[4], Theta_dot[5]]).T # the actual state




            client_socket.send(input_str.encode())


            # Receive the result from the server
            result = client_socket.recv(4096).decode()  # Increase buffer size if necessary

            # Split the result into rows for the mass matrix
            rows = result.split('\n')

            # Convert the received matrix rows to a numpy array
            M = np.array([list(map(float, row.split(','))) for row in rows])

            # the minimum and maximum eigenvalues of the P matrix
            m_1 = 4.828198042151454e+01
            m_2 = max(self.kp.values())

            # the P matrix for the Lyapunov function
            P = block_diag(self.KpMatrix, M)
            P = 0.5 * P

            q_v1_0 = th_v0[0]
            q_v2_0 = th_v0[1] + math.radians(7.869116397416903e+01)
            q_v3_0 = th_v0[2] +  math.radians(1.130769572634749e+01)
            q_v5_0 = th_v0[3]


            # use the neural network to predict the threshold and then compute the DSM
            gamma  = self.predict(q_v1_0, q_v2_0, q_v3_0, q_v5_0) - self.Tolerance

            Lyapunov = (X-X_v).T @ P @ (X-X_v)

            DSM = max(gamma - Lyapunov, 0)



            self.get_logger().info(f'gamma Value: {gamma}')

            self.get_logger().info(f'Lyapunov Function Value: {Lyapunov}')


            # the NF is computed in the Cartesian space
            # NF diffeomorfsim
            vx_0, vy_0, vz_0 = forward_kinematics(np.rad2deg(q_v1_0), np.rad2deg(q_v2_0), np.rad2deg(q_v3_0), 0.0, np.rad2deg(q_v5_0), 0.0) #Cartesian

            v_0 = np.array([ vx_0, vy_0, vz_0]).T #Cartesian
            self.get_logger().info(f'v_0 : {v_0}')

            
            rho_a = (self.r - v_0) / max(np.linalg.norm(self.r - v_0),self.eta1) #Cartesian

            c1 = v_0[1] - self.f1(v_0[0])
            c2 = self.f2(v_0[0]) - v_0[1]



            rho_r1 = max(0, (self.zeta - c1) / (self.zeta - self.delta)) * np.array([
                (-self.Beta * (self.a - self.k1) * (v_0[0] - self.b)) / max(np.linalg.norm(-self.Beta * (self.a - self.k1) * (v_0[0] - self.b)), 0.01),
                1 / np.linalg.norm(1),
                0
            ]).T


            rho_r2 = max(0, (self.zeta - c2) / (self.zeta - self.delta)) * np.array([
                (self.Beta * (self.a - self.k2) * (v_0[0] - self.b)) / max(np.linalg.norm(self.Beta * (self.a - self.k2) * (v_0[0] - self.b)), 0.01),
                -1 / np.linalg.norm(-1),
                0
            ]).T

            rho_r = rho_r1 + rho_r2

            Attraction_Field = rho_a + rho_r

            g = DSM * Attraction_Field


            # the equations for computing dynamic kappa
            c = min(c1,c2)
            vartheta = c - self.delta

            kappa_val  = (max((np.sqrt(m_1) / (np.sqrt(m_1) + np.sqrt(m_2))) * vartheta - (np.sqrt(m_2) / (np.sqrt(m_1) + np.sqrt(m_2))) * (np.linalg.norm(X - X_v)), 0)) / (self.mu * self.dt_v * max(np.linalg.norm(g), self.eta2))  # dynamic kappa

            # update the applied reference
            self.kappa.append(kappa_val)
            v_dot = self.kappa[-1] * g # 
            self.v.append(v_0 + v_dot * self.dt_v)
            self.get_logger().info(f'v : {self.v[-1]}')

        else:
            # diffeomorfsim inverse 
            q_v1, q_v2, q_v3, q_v4, q_v5, q_v6 = inverse_kinematics(self.v[-1][0], self.v[-1][1], self.v[-1][2])


            self.th_v = np.array([math.radians(q_v1), math.radians(q_v2) - math.radians(7.869116397416903e+01), math.radians(q_v3) - math.radians(1.130769572634749e+01), q_v4, math.radians(q_v5), q_v6]).T


            X_v = np.array([self.th_v[0], self.th_v[1], self.th_v[2], self.th4_d, self.th_v[4], self.th6_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T  #the equilibrium state
            X = np.array([Theta[0], Theta[1], Theta[2], Theta[3], Theta[4], Theta[5], Theta_dot[0], Theta_dot[1], Theta_dot[2], Theta_dot[3], Theta_dot[4], Theta_dot[5]]).T




            client_socket.send(input_str.encode())


            # Receive the result from the server
            result = client_socket.recv(4096).decode()  # Increase buffer size if necessary

            # Split the result into rows for the mass matrix
            rows = result.split('\n')

            # Convert the received matrix rows to a numpy array
            M = np.array([list(map(float, row.split(','))) for row in rows])


            # the minimum and maximum eigenvalues of the P matrix
            m_1 = 4.828198042151454e+01
            m_2 = max(self.kp.values())

            # the P matrix for the Lyapunov function
            P = block_diag(self.KpMatrix, M)
            P = 0.5 * P


           
            # Compute Lyapunov function           
            Lyapunov = (X-X_v).T @ P @ (X-X_v)
            self.get_logger().info(f'Lyapunov Function Value: {Lyapunov}')



            # use the neural network to predict the threshold and then compute the DSM
            # the neural network was trained with training inputs as angles in radians

            gamma  = self.predict(math.radians(q_v1), math.radians(q_v2), math.radians(q_v3), math.radians(q_v5)) - self.Tolerance


            self.get_logger().info(f'gamma Value: {gamma}')

 




            DSM = max(gamma - Lyapunov, 0)
            self.get_logger().info(f'DSM: {DSM}')


            # the NF is computed in the Cartesian space
            # NF diffeomorfsim

            rho_a = (self.r - self.v[-1]) / max(np.linalg.norm(self.r - self.v[-1]),self.eta1)  #Cartesian

            c1 = self.v[-1][1] - self.f1(self.v[-1][0])
            c2 = self.f2(self.v[-1][0]) - self.v[-1][1]


            rho_r1 = max(0, (self.zeta - c1) / (self.zeta - self.delta)) * np.array([
                (-self.Beta * (self.a - self.k1) * (self.v[-1][0] - self.b)) / max(np.linalg.norm(-self.Beta * (self.a - self.k1) * (self.v[-1][0] - self.b)), 0.01),
                1 / np.linalg.norm(1),
                0
            ]).T


            rho_r2 = max(0, (self.zeta - c2) / (self.zeta - self.delta)) * np.array([
                (self.Beta * (self.a - self.k2) * (self.v[-1][0] - self.b)) / max(np.linalg.norm(self.Beta * (self.a - self.k2) * (self.v[-1][0] - self.b)), 0.01),
                -1 / np.linalg.norm(-1),
                0
            ]).T

            rho_r = rho_r1 + rho_r2


            Attraction_Field = rho_a + rho_r

            g = DSM * Attraction_Field




            # the equations for computing dynamic kappa
            c = min(c1,c2)
            vartheta = c - self.delta




            kappa_val = (max((np.sqrt(m_1) / (np.sqrt(m_1) + np.sqrt(m_2))) * vartheta - (np.sqrt(m_2) / (np.sqrt(m_1) + np.sqrt(m_2))) * (np.linalg.norm(X - X_v)), 0)) / (self.mu * self.dt_v * max(np.linalg.norm(g), self.eta2))  # dynamic kappa

            # update the applied reference
            self.kappa.append(kappa_val)
            v_dot = self.kappa[-1] * g # -1 refers to the last element
            self.v.append(self.v[-1] + v_dot * self.dt_v)
            self.get_logger().info(f'v : {self.v[-1]}')


        


        client_socket.close()




    # the wall constraints
    def f1(self, x):
        return (0.5 * (self.a - self.k1)) * self.Beta * (x - self.b)**2 + 0.5 * (self.a + self.k1)

    def f2(self, x):
        return (0.5 * (self.a - self.k2)) * self.Beta * (x - self.b)**2 + 0.5 * (self.a + self.k2)




    # ---------------------------
    # Function for inference
    # ---------------------------
    def predict(self,q_v1, q_v2, q_v3, q_v5):
        """ Predict the output given four joint angles """
        


        # Prepare the input data (1 sample with 4 features)
        inputs = np.array([[q_v1, q_v2, q_v3, q_v5]], dtype=np.float32)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            predicted_output = model(inputs_tensor)

        # Convert to NumPy for easy use
        result = predicted_output.cpu().numpy().flatten()[0]
        
        return result







    def save_data(self):
        with open('ERG_2_data_constantKappa2_4review_test9.json', 'w') as f:
            json.dump(self.joint_data, f, indent=4)
        self.get_logger().info('Joint data saved.json')











def main(args=None):
    rclpy.init(args=args)
    viperx_300s_controller = ViperX300sController()
    rclpy.spin(viperx_300s_controller)

    viperx_300s_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





















































