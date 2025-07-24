import numpy as np
# this function gives the I.K solution to a given x,y,z position. The provided solution is based on which the desired end effector pose to be always pointing towards the x,y,z position
def inverse_kinematics(px, py, pz):
    d6 = 70 + 68.7 + 67.88  # distance from frame 6 to end effector
    
    # Desired attitude of the end effector
    th = np.arctan2(py, px)
    R06 = np.array([[0, np.sin(th), np.cos(th)],
                    [0, -np.cos(th), np.sin(th)],
                    [1, 0, 0]])
    Oc = np.array([px, py, pz]) - R06 @ np.array([0, 0, d6])
    pxc, pyc, pzc = Oc
    
    q1 = np.rad2deg(np.arctan2(pyc, pxc))
    
    T01inv = np.array([[np.cos(np.deg2rad(q1)), np.sin(np.deg2rad(q1)), 0, 0],
                       [-np.sin(np.deg2rad(q1)), np.cos(np.deg2rad(q1)), 0, 0],
                       [0, 0, 1, -126.75],
                       [0, 0, 0, 1]])
    P06 = np.array([pxc, pyc, pzc, 1])
    P16 = T01inv @ P06
    
    beta1 = np.rad2deg(np.arctan2(P16[2], P16[0]))
    beta2 = np.rad2deg(np.arccos((305.94**2 - 300**2 + np.linalg.norm(P16[:3])**2) / (2 * 305.94 * np.linalg.norm(P16[:3]))))
    phi = np.rad2deg(np.arccos((300**2 + 305.94**2 - np.linalg.norm(P16[:3])**2) / (2 * 300 * 305.94)))
    
    if abs(beta1) > 78.69116397416903:
        s = phi - 90
        theta3 = 180 + s - 360
        q3 = theta3 + 11.30769572634749
        theta2 = -(beta1 - beta2)
        q2 = theta2 + 78.69116397416903
    else:
        s = 270 - phi
        theta3 = 180 + s
        q3 = theta3 + 11.30769572634749 - 360
        theta2 = -(beta1 + beta2)
        q2 = theta2 + 78.69116397416903
    
    theta = np.deg2rad([q1, q2 - 78.69116397416903, q3 - 11.30769572634749])
    alpha = np.deg2rad([0, 270, 0, 270, 90, 270, 0])
    a = [0, 0, 305.94, 0, 0, 0, 0]
    d = [126.75, 0, 0, 300, 0, 0, 68.7 + 70]
    


    # Define transformation matrices
    T01 = np.array([
        [np.cos(theta[0]), -np.sin(theta[0]), 0, a[0]],
        [np.sin(theta[0]) * np.cos(alpha[0]), np.cos(theta[0]) * np.cos(alpha[0]), -np.sin(alpha[0]), -d[0] * np.sin(alpha[0])],
        [np.sin(theta[0]) * np.sin(alpha[0]), np.cos(theta[0]) * np.sin(alpha[0]), np.cos(alpha[0]), d[0] * np.cos(alpha[0])],
        [0, 0, 0, 1]
    ])

    T12 = np.array([
        [np.cos(theta[1]), -np.sin(theta[1]), 0, a[1]],
        [np.sin(theta[1]) * np.cos(alpha[1]), np.cos(theta[1]) * np.cos(alpha[1]), -np.sin(alpha[1]), -d[1] * np.sin(alpha[1])],
        [np.sin(theta[1]) * np.sin(alpha[1]), np.cos(theta[1]) * np.sin(alpha[1]), np.cos(alpha[1]), d[1] * np.cos(alpha[1])],
        [0, 0, 0, 1]
    ])

    T23 = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]), 0, a[2]],
        [np.sin(theta[2]) * np.cos(alpha[2]), np.cos(theta[2]) * np.cos(alpha[2]), -np.sin(alpha[2]), -d[2] * np.sin(alpha[2])],
        [np.sin(theta[2]) * np.sin(alpha[2]), np.cos(theta[2]) * np.sin(alpha[2]), np.cos(alpha[2]), d[2] * np.cos(alpha[2])],
        [0, 0, 0, 1]
    ])




    T03 = np.dot(np.dot(T01, T12), T23)

    R03 = T03[:3, :3]


    R36 = R03.T @ R06
    
    q4_sol1 = np.rad2deg(np.arctan2(R36[2, 2], -R36[0, 2]))
    q5_sol1 = np.rad2deg(np.arctan2(np.sqrt(R36[0, 2]**2 + R36[2, 2]**2), R36[1, 2]))
    q6_sol1 = np.rad2deg(np.arctan2(-R36[1, 1], R36[1, 0]))
    
    q4_sol2 = np.rad2deg(np.arctan2(-R36[2, 2], R36[0, 2]))
    q5_sol2 = np.rad2deg(np.arctan2(-np.sqrt(R36[0, 2]**2 + R36[2, 2]**2), R36[1, 2]))
    q6_sol2 = np.rad2deg(np.arctan2(-R36[1, 1], -R36[1, 0]))
    
    solutions_q5 = [q5_sol1, q5_sol2]
    q4, q6 = 0, 0
    best_error = np.inf
    
    for sol_q5 in solutions_q5:
        theta = np.deg2rad([q1, q2 - 78.69116397416903, q3 - 11.30769572634749, q4, sol_q5, q6])
        
        
        T34 = np.array([
            [np.cos(theta[3]), -np.sin(theta[3]), 0, a[3]],
            [np.sin(theta[3]) * np.cos(alpha[3]), np.cos(theta[3]) * np.cos(alpha[3]), -np.sin(alpha[3]), -d[3] * np.sin(alpha[3])],
            [np.sin(theta[3]) * np.sin(alpha[3]), np.cos(theta[3]) * np.sin(alpha[3]), np.cos(alpha[3]), d[3] * np.cos(alpha[3])],
            [0, 0, 0, 1]
        ])

        T45 = np.array([
            [np.cos(theta[4]), -np.sin(theta[4]), 0, a[4]],
            [np.sin(theta[4]) * np.cos(alpha[4]), np.cos(theta[4]) * np.cos(alpha[4]), -np.sin(alpha[4]), -d[4] * np.sin(alpha[4])],
            [np.sin(theta[4]) * np.sin(alpha[4]), np.cos(theta[4]) * np.sin(alpha[4]), np.cos(alpha[4]), d[4] * np.cos(alpha[4])],
            [0, 0, 0, 1]
        ])

        T56 = np.array([
            [np.cos(theta[5]), -np.sin(theta[5]), 0, a[5]],
            [np.sin(theta[5]) * np.cos(alpha[5]), np.cos(theta[5]) * np.cos(alpha[5]), -np.sin(alpha[5]), -d[5] * np.sin(alpha[5])],
            [np.sin(theta[5]) * np.sin(alpha[5]), np.cos(theta[5]) * np.sin(alpha[5]), np.cos(alpha[5]), d[5] * np.cos(alpha[5])],
            [0, 0, 0, 1]
        ])



        # Calculate T06
        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

        # Extract rotation matrix R06 and position vector
        R06_star = T06[:3, :3]
        position = T06[:3, 3] + R06_star @ np.array([0, 0, d6])





        total_error = np.linalg.norm(position - [px, py, pz])
        if total_error < best_error:
            best_error = total_error
            q5 = sol_q5
    
    return q1, q2, q3, q4, q5, q6
