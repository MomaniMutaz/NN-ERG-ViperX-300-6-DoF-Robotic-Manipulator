import numpy as np
# this function gives the I.K solution to a given x,y,z position. The provided solution is based on which the desired end effector pose to be always pointing towards the x,y,z position
def forward_kinematics(q1, q2, q3, q4, q5, q6):
    d6 = 70 + 68.7 + 67.88  # distance from frame 6 to end effector
    

    
    theta = np.deg2rad([q1, q2 - 78.69116397416903, q3 - 11.30769572634749, q4, q5, q6])
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

    px, py, pz = position[:3]


    return px, py, pz
