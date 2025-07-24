import numpy as np




def calculate_gravity(q1,q2,q3,q4,q5,q6):


    # % Gravity_Compensationfunction: Computes the gravity compensation vector for given joint angles.



    # % Note: The model derivation used standard DH parameters with the following
    # % theta values: theta = {q1, q2, q3, q4, q5, q6}.
    # %
    # % However, to simplify parameter extraction by using pure equations, we
    # % initially omitted bias terms in q2 and q3. These biases are:
    # % - q2 offset by 78.69 degrees
    # % - q3 offset by 11.31 degrees
    # %
    # % Here, we adjust q2 and q3 by their respective offsets before computing the matrix:




    # Define lengths in meters
    L1 = 126.75 / 1000
    L2 = 305.94 / 1000
    L3 = (300 - 103.62) / 1000
    L4 = (103.62) / 1000
    L5 = 70 / 1000
    L6 = (68.7 + 67.88) / 1000

    # Define gravitational acceleration
    g = 9.81

    # Adjust joint angles (q2 and q3) in radians
    q2 = q2 - np.deg2rad(78.69)
    q3 = q3 - np.deg2rad(11.31)




    # % The identified base parameters (\vartheta_b vector). 

    # % These parameters were identified using applied torque data measured in milliAmps, 
    # % as the manipulator operates in current control mode. In current control, the control 
    # % law generates the control signal as a current (milliAmps). To achieve stable operation, 
    # % we pre-stabilized the system with a PD control law combined with gravity compensation (PD+G).
    # % 
    # % This gravity compensation vector needs to produce a current-based signal, so it is essential 
    # % to use dynamical parameters identified from current-based torque measurements in the 
    # % gravity compensation function. 
    # % 
    # % Consequently, when using a mass matrix in model-based control, it is crucial that the matrix 
    # % also relies on parameters identified with torque data measured in currents. This ensures 
    # % consistency between the control law, gravity compensation, and identified model parameters.
 





    coefficients =np.array([

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
    -1.700949877698571e+01
    ])

    #   % The identified dynamical parameters (\vartheta vector). 

    PhysicalParameter =np.array([

    5.616837270769171e+01,
     5.821926829510207e+01,
     5.854915317439252e+01,
     5.746741189289997e+01,
     5.385749121571954e+01,
     6.400185300213873e+00,
     6.400185294426747e+00,
     5.932935135307417e+01,
     5.553107989408500e+01,
     5.302565278048813e+01,
     1.021225564960032e-01,
    -1.021225564961166e-01,
     4.426055992816626e+01,
    -3.314754420087764e-05,
     4.247166603154148e+01,
     5.011134465169280e+01,
     2.300006342119864e+01,
     2.300006342061333e+01,
     2.300006342094709e+01,
     3.100003428025688e+01,
     5.416010659639789e+01,
     5.211115313844498e+01,
     2.200014252683767e+01,
    -3.938018598862479e+00,
     3.417781523371926e+00,
     5.023875885969111e+01,
     6.600004360521282e+01,
     4.173267622114748e+01,
     3.831713267536631e+00,
     2.811215335388378e+01,
    -4.003428466941468e+00,
     2.840960888110710e+01,
    -4.226307201152052e+00,
     2.780412234882357e+01,
    -6.936788332855680e-01,
     2.784248558484121e+01,
    -3.721691902190042e-01,
     2.822410614312182e+01,
    -7.238960740607910e-01,
     2.827392409499520e+01,
     5.149496691573201e+00,
     2.666947182641271e+01,
    -1.107793944122596e+00,
     2.915595659791496e+01,
     4.296533548421897e+00,
     2.897675542365281e+01,
     2.379492119538459e+00,
     3.293370232816496e+01,
    -1.613096253820716e+00,
     3.049211239036838e+01,
     9.695318327507249e-01,
     3.033481471882235e+01,
     3.674494590695232e+01,
    -2.706112421160014e-01,
     3.467180898132277e+01,
     3.551758865734764e+00,
     3.562104816245145e+01,
    -9.429655287462220e-01,
     1.773660600397160e+00,
    -8.178635773950234e-01,
     4.300168626372800e-01,
     1.100902104191545e-01,
    -4.162646835747426e-01,
     5.622553300896489e-01,
     5.015703062523544e-01,
    -2.499340998364314e-01,
    -1.857520657262411e-01,
     6.660621761258031e-03,
    -2.940184087539479e-01,
     3.345802928060783e-01,
    -3.294505520801637e-01,
    -4.123384175096043e-01,
     5.539813861979815e+01
    ])







    # % Extract the dynamical parameters, noting that the inertia terms (I elements) are 
    # % calculated around each link's center of mass (COM). Additional terms like mX^2, mY^2, 
    # % , mZ^2, mxy, myz, and mxz appear due to the parallel axis theorem, which accounts for the offset 
    # % of each link's mass relative to the axis of rotation.

    Ixx6 = PhysicalParameter[0]
    Iyy6 = PhysicalParameter[1]
    Ixx5 = PhysicalParameter[2]
    Iyy5 = PhysicalParameter[3]
    Izz5 = PhysicalParameter[4]
    m5Y5 = PhysicalParameter[5]
    m6Z6 = PhysicalParameter[6]
    Ixx4 = PhysicalParameter[7]
    Iyy4 = PhysicalParameter[8]
    Izz4 = PhysicalParameter[9]
    m4Y4 = PhysicalParameter[10]
    m5Z5 = PhysicalParameter[11]
    Ixx3 = PhysicalParameter[12]
    m4Z4 = PhysicalParameter[13]
    Iyy3 = PhysicalParameter[14]
    Izz3 = PhysicalParameter[15]
    m6 = PhysicalParameter[16]
    m5 = PhysicalParameter[17]
    m4 = PhysicalParameter[18]
    m3Y3 = PhysicalParameter[19]
    Ixx2 = PhysicalParameter[20]
    Iyy2 = PhysicalParameter[21]
    m3 = PhysicalParameter[22]
    Ixz2 = PhysicalParameter[23]
    m3Z3 = PhysicalParameter[24]
    Izz2 = PhysicalParameter[25]
    m2X2 = PhysicalParameter[26]
    Izz1 = PhysicalParameter[27]

    Ixy6 = PhysicalParameter[28]
    m6Z6_sqrd = PhysicalParameter[29]
    Ixz6 = PhysicalParameter[30]
    m6Y6_sqrd = PhysicalParameter[31]
    Iyz6 = PhysicalParameter[32]
    m6X6_sqrd = PhysicalParameter[33]

    Ixy5 = PhysicalParameter[34]
    m5Z5_sqrd = PhysicalParameter[35]
    Ixz5 = PhysicalParameter[36]
    m5Y5_sqrd = PhysicalParameter[37]
    Iyz5 = PhysicalParameter[38]
    m5X5_sqrd = PhysicalParameter[39]

    Ixy4 = PhysicalParameter[40]
    m4Z4_sqrd = PhysicalParameter[41]
    Ixz4 = PhysicalParameter[42]
    m4Y4_sqrd = PhysicalParameter[43]
    Iyz4 = PhysicalParameter[44]
    m4X4_sqrd = PhysicalParameter[45]

    Ixy3 = PhysicalParameter[46]
    m3Z3_sqrd = PhysicalParameter[47]
    Ixz3 = PhysicalParameter[48]
    m3Y3_sqrd = PhysicalParameter[49]
    Iyz3 = PhysicalParameter[50]
    m3X3_sqrd = PhysicalParameter[51]

    m2Y2_sqrd = PhysicalParameter[52]
    Ixy2 = PhysicalParameter[53]
    m2Z2_sqrd = PhysicalParameter[54]
    Iyz2 = PhysicalParameter[55]
    m2X2_sqrd = PhysicalParameter[56]

    m2X2Y2 = PhysicalParameter[57]
    m2Y2Z2 = PhysicalParameter[58]
    m2X2Z2 = PhysicalParameter[59]
    m3X3Y3 = PhysicalParameter[60]
    m3Y3Z3 = PhysicalParameter[61]
    m3X3Z3 = PhysicalParameter[62]
    m4X4Y4 = PhysicalParameter[63]
    m4Y4Z4 = PhysicalParameter[64]
    m4X4Z4 = PhysicalParameter[65]
    m5X5Y5 = PhysicalParameter[66]
    m5Y5Z5 = PhysicalParameter[67]
    m5X5Z5 = PhysicalParameter[68]
    m6X6Y6 = PhysicalParameter[69]
    m6Y6Z6 = PhysicalParameter[70]
    m6X6Z6 = PhysicalParameter[71]
    Izz6 = PhysicalParameter[72]






    IzzR1 = coefficients[0]

    IxxR2 = coefficients[1]
    Jxy2 = coefficients[2]
    IxzR2 = coefficients[3]
    Jyz2 = coefficients[4]
    IzzR2 = coefficients[5]
    m2XR2 = coefficients[6]
    m2Y2 = coefficients[7]

    IxxR3 = coefficients[8]
    Jxy3 = coefficients[9]
    Jxz3 = coefficients[10]
    Jyz3 = coefficients[11]
    IzzR3 = coefficients[12]
    m3X3 = coefficients[13]
    m3YR3 = coefficients[14]

    IxxR4 = coefficients[15]
    Jxy4 = coefficients[16]
    Jxz4 = coefficients[17]
    Jyz4 = coefficients[18]
    IzzR4 = coefficients[19]
    m4X4 = coefficients[20]
    m4YR4 = coefficients[21]

    IxxR5 = coefficients[22]
    Jxy5 = coefficients[23]
    Jxz5 = coefficients[24]
    Jyz5 = coefficients[25]
    IzzR5 = coefficients[26]
    m5X5 = coefficients[27]
    m5YR5 = coefficients[28]

    IxxR6 = coefficients[29]
    Jxy6 = coefficients[30]
    Jxz6 = coefficients[31]
    Jyz6 = coefficients[32]
    Jzz6 = coefficients[33]
    m6X6 = coefficients[34]
    m6Y6 = coefficients[35]







    # % In Mathematica, known parameter values (e.g., lengths and gravity) were left as variables,
    # % rather than plugging in numerical values. This approach allowed us to construct the 
    # % regression matrix symbolically and extract components M(q), V(q, q_dot), and G(q) more easily. 
    # %
    # % We defined symbolic terms (e.g., cof1, cof2) for combinations of these parameters.
    # % After identifying parameter values, we can substitute them back into the model. 
    # % Below, cof# terms represent combinations of gravitational acceleration (g), masses (m#), 
    # % link lengths (L#), and first moment of masses (m#X#, m#Y#, m#Z#). 

    # % link lengths (L#) were not identified, they are given by the
    # % manufacturer.

    # % Define cof# terms using symbolic combinations of parameters:




    cof1 = g * m2X2
    cof2 = g * m2Y2
    cof3 = L2**2 * m3
    cof4 = g * L2 * m3
    cof5 = L2 * m3X3
    cof6 = g * m3X3
    cof7 = L2 * m3Y3
    cof8 = g * m3Y3
    cof9 = L2 * m3Z3
    cof10 = L2**2 * m4
    cof11 = L2 * L3 * m4
    cof12 = L2 * L4 * m4
    cof13 = L3**2 * m4
    cof14 = L3 * L4 * m4
    cof15 = L4**2 * m4
    cof16 = g * L3 * m4
    cof17 = g * L4 * m4
    cof18 = g * L2 * m4
    cof19 = L2 * m4X4
    cof20 = L3 * m4X4
    cof21 = L4 * m4X4
    cof22 = g * m4X4
    cof23 = L2 * m4Y4
    cof24 = L3 * m4Y4
    cof25 = L4 * m4Y4
    cof26 = g * m4Y4
    cof27 = L2 * m4Z4
    cof28 = L3 * m4Z4
    cof29 = L4 * m4Z4
    cof30 = g * m4Z4
    cof31 = L2**2 * m5
    cof32 = L2 * L3 * m5
    cof33 = L2 * L4 * m5
    cof34 = L3 * L4 * m5
    cof35 = L3**2 * m5
    cof36 = L4**2 * m5
    cof37 = g * L3 * m5
    cof38 = g * L4 * m5
    cof39 = g * L2 * m5
    cof40 = L2 * m5X5
    cof41 = L3 * m5X5
    cof42 = L4 * m5X5
    cof43 = g * m5X5
    cof44 = L2 * m5Y5
    cof45 = L3 * m5Y5
    cof46 = L4 * m5Y5
    cof47 = g * m5Y5
    cof48 = L2 * m5Z5
    cof49 = L3 * m5Z5
    cof50 = L4 * m5Z5
    cof51 = g * m5Z5
    cof52 = L2**2 * m6
    cof53 = L2 * L4 * m6
    cof54 = L2 * L3 * m6
    cof55 = L3**2 * m6
    cof56 = L4**2 * m6
    cof57 = L3 * L4 * m6
    cof58 = g * L3 * m6
    cof59 = g * L4 * m6
    cof60 = g * L2 * m6
    cof61 = L2 * m6X6
    cof62 = L3 * m6X6
    cof63 = L4 * m6X6
    cof64 = g * m6X6
    cof65 = L2 * m6Y6
    cof66 = L3 * m6Y6
    cof67 = L4 * m6Y6
    cof68 = g * m6Y6
    cof69 = L2 * m6Z6
    cof70 = L3 * m6Z6
    cof71 = L4 * m6Z6
    cof72 = g * m6Z6

    cof73 = m2X2_sqrd
    cof74 = m2Y2_sqrd
    cof75 = m2Z2_sqrd
    cof76 = m3X3_sqrd
    cof77 = m3Y3_sqrd
    cof78 = m3Z3_sqrd
    cof79 = m4X4_sqrd
    cof80 = m4Y4_sqrd
    cof81 = m4Z4_sqrd
    cof82 = m5X5_sqrd
    cof83 = m5Y5_sqrd
    cof84 = m5Z5_sqrd
    cof85 = m6X6_sqrd
    cof86 = m6Y6_sqrd
    cof87 = m6Z6_sqrd


    cof88 = m2X2Y2   
    cof89 = m2Y2Z2   
    cof90 = m2X2Z2   

    cof91 = m3X3Y3   
    cof92 = m3Y3Z3   
    cof93 = m3X3Z3   

    cof94 = m4X4Y4   
    cof95 = m4Y4Z4   
    cof96 = m4X4Z4   

    cof97 = m5X5Y5   
    cof98 = m5Y5Z5   
    cof99 = m5X5Z5   

    cof100 = m6X6Y6  
    cof101 = m6Y6Z6  
    cof102 = m6X6Z6  












    G1 = 0.0









    G2 = (-1) * cof1 * np.cos(q2) \
    - cof18 * np.cos(q2) \
    - cof39 * np.cos(q2) \
    - cof4 * np.cos(q2) \
    - cof60 * np.cos(q2) \
    - cof6 * np.cos(q2) * np.cos(q3) \
    - cof22 * np.cos(q2) * np.cos(q3) * np.cos(q4) \
    - cof43 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.cos(q5) \
    - cof64 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.cos(q6) \
    + cof2 * np.sin(q2) \
    + cof16 * np.cos(q3) * np.sin(q2) \
    + cof17 * np.cos(q3) * np.sin(q2) \
    + cof30 * np.cos(q3) * np.sin(q2) \
    + cof37 * np.cos(q3) * np.sin(q2) \
    + cof38 * np.cos(q3) * np.sin(q2) \
    + cof58 * np.cos(q3) * np.sin(q2) \
    + cof59 * np.cos(q3) * np.sin(q2) \
    + cof8 * np.cos(q3) * np.sin(q2) \
    + cof47 * np.cos(q3) * np.cos(q5) * np.sin(q2) \
    + cof72 * np.cos(q3) * np.cos(q5) * np.sin(q2) \
    + cof16 * np.cos(q2) * np.sin(q3) \
    + cof17 * np.cos(q2) * np.sin(q3) \
    + cof30 * np.cos(q2) * np.sin(q3) \
    + cof37 * np.cos(q2) * np.sin(q3) \
    + cof38 * np.cos(q2) * np.sin(q3) \
    + cof58 * np.cos(q2) * np.sin(q3) \
    + cof59 * np.cos(q2) * np.sin(q3) \
    + cof8 * np.cos(q2) * np.sin(q3) \
    + cof47 * np.cos(q2) * np.cos(q5) * np.sin(q3) \
    + cof72 * np.cos(q2) * np.cos(q5) * np.sin(q3) \
    + cof6 * np.sin(q2) * np.sin(q3) \
    + cof22 * np.cos(q4) * np.sin(q2) * np.sin(q3) \
    + cof43 * np.cos(q4) * np.cos(q5) * np.sin(q2) * np.sin(q3) \
    + cof64 * np.cos(q4) * np.cos(q5) * np.cos(q6) * np.sin(q2) * np.sin(q3) \
    + cof26 * np.cos(q2) * np.cos(q3) * np.sin(q4) \
    - cof51 * np.cos(q2) * np.cos(q3) * np.sin(q4) \
    + cof68 * np.cos(q2) * np.cos(q3) * np.cos(q6) * np.sin(q4) \
    - cof26 * np.sin(q2) * np.sin(q3) * np.sin(q4) \
    + cof51 * np.sin(q2) * np.sin(q3) * np.sin(q4) \
    - cof68 * np.cos(q6) * np.sin(q2) * np.sin(q3) * np.sin(q4) \
    + cof47 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q5) \
    + cof72 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q5) \
    + cof43 * np.cos(q3) * np.sin(q2) * np.sin(q5) \
    + cof64 * np.cos(q3) * np.cos(q6) * np.sin(q2) * np.sin(q5) \
    + cof43 * np.cos(q2) * np.sin(q3) * np.sin(q5) \
    + cof64 * np.cos(q2) * np.cos(q6) * np.sin(q3) * np.sin(q5) \
    - cof47 * np.cos(q4) * np.sin(q2) * np.sin(q3) * np.sin(q5) \
    - cof72 * np.cos(q4) * np.sin(q2) * np.sin(q3) * np.sin(q5) \
    + cof68 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.sin(q6) \
    - cof68 * np.cos(q4) * np.cos(q5) * np.sin(q2) * np.sin(q3) * np.sin(q6) \
    + cof64 * np.cos(q2) * np.cos(q3) * np.sin(q4) * np.sin(q6) \
    - cof64 * np.sin(q2) * np.sin(q3) * np.sin(q4) * np.sin(q6) \
    - cof68 * np.cos(q3) * np.sin(q2) * np.sin(q5) * np.sin(q6) \
    - cof68 * np.cos(q2) * np.sin(q3) * np.sin(q5) * np.sin(q6)











    G3 = (
    -cof6 * np.cos(q2) * np.cos(q3)
    - cof22 * np.cos(q2) * np.cos(q3) * np.cos(q4)
    - cof43 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.cos(q5)
    - cof64 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.cos(q6)
    + cof16 * np.cos(q3) * np.sin(q2)
    + cof17 * np.cos(q3) * np.sin(q2)
    + cof30 * np.cos(q3) * np.sin(q2)
    + cof37 * np.cos(q3) * np.sin(q2)
    + cof38 * np.cos(q3) * np.sin(q2)
    + cof58 * np.cos(q3) * np.sin(q2)
    + cof59 * np.cos(q3) * np.sin(q2)
    + cof8 * np.cos(q3) * np.sin(q2)
    + cof47 * np.cos(q3) * np.cos(q5) * np.sin(q2)
    + cof72 * np.cos(q3) * np.cos(q5) * np.sin(q2)
    + cof16 * np.cos(q2) * np.sin(q3)
    + cof17 * np.cos(q2) * np.sin(q3)
    + cof30 * np.cos(q2) * np.sin(q3)
    + cof37 * np.cos(q2) * np.sin(q3)
    + cof38 * np.cos(q2) * np.sin(q3)
    + cof58 * np.cos(q2) * np.sin(q3)
    + cof59 * np.cos(q2) * np.sin(q3)
    + cof8 * np.cos(q2) * np.sin(q3)
    + cof47 * np.cos(q2) * np.cos(q5) * np.sin(q3)
    + cof72 * np.cos(q2) * np.cos(q5) * np.sin(q3)
    + cof6 * np.sin(q2) * np.sin(q3)
    + cof22 * np.cos(q4) * np.sin(q2) * np.sin(q3)
    + cof43 * np.cos(q4) * np.cos(q5) * np.sin(q2) * np.sin(q3)
    + cof64 * np.cos(q4) * np.cos(q5) * np.cos(q6) * np.sin(q2) * np.sin(q3)
    + cof26 * np.cos(q2) * np.cos(q3) * np.sin(q4)
    - cof51 * np.cos(q2) * np.cos(q3) * np.sin(q4)
    + cof68 * np.cos(q2) * np.cos(q3) * np.cos(q6) * np.sin(q4)
    - cof26 * np.sin(q2) * np.sin(q3) * np.sin(q4)
    + cof51 * np.sin(q2) * np.sin(q3) * np.sin(q4)
    - cof68 * np.cos(q6) * np.sin(q2) * np.sin(q3) * np.sin(q4)
    + cof47 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q5)
    + cof72 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q5)
    + cof43 * np.cos(q3) * np.sin(q2) * np.sin(q5)
    + cof64 * np.cos(q3) * np.cos(q6) * np.sin(q2) * np.sin(q5)
    + cof43 * np.cos(q2) * np.sin(q3) * np.sin(q5)
    + cof64 * np.cos(q2) * np.cos(q6) * np.sin(q3) * np.sin(q5)
    - cof47 * np.cos(q4) * np.sin(q2) * np.sin(q3) * np.sin(q5)
    - cof72 * np.cos(q4) * np.sin(q2) * np.sin(q3) * np.sin(q5)
    + cof68 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.sin(q6)
    - cof68 * np.cos(q4) * np.cos(q5) * np.sin(q2) * np.sin(q3) * np.sin(q6)
    + cof64 * np.cos(q2) * np.cos(q3) * np.sin(q4) * np.sin(q6)
    - cof64 * np.sin(q2) * np.sin(q3) * np.sin(q4) * np.sin(q6)
    - cof68 * np.cos(q3) * np.sin(q2) * np.sin(q5) * np.sin(q6)
    - cof68 * np.cos(q2) * np.sin(q3) * np.sin(q5) * np.sin(q6)
)








    G4 = (
    cof26 * np.cos(q3) * np.cos(q4) * np.sin(q2)
    - cof51 * np.cos(q3) * np.cos(q4) * np.sin(q2)
    + cof68 * np.cos(q3) * np.cos(q4) * np.cos(q6) * np.sin(q2)
    + cof26 * np.cos(q2) * np.cos(q4) * np.sin(q3)
    - cof51 * np.cos(q2) * np.cos(q4) * np.sin(q3)
    + cof68 * np.cos(q2) * np.cos(q4) * np.cos(q6) * np.sin(q3)
    + cof22 * np.cos(q3) * np.sin(q2) * np.sin(q4)
    + cof43 * np.cos(q3) * np.cos(q5) * np.sin(q2) * np.sin(q4)
    + cof64 * np.cos(q3) * np.cos(q5) * np.cos(q6) * np.sin(q2) * np.sin(q4)
    + cof22 * np.cos(q2) * np.sin(q3) * np.sin(q4)
    + cof43 * np.cos(q2) * np.cos(q5) * np.sin(q3) * np.sin(q4)
    + cof64 * np.cos(q2) * np.cos(q5) * np.cos(q6) * np.sin(q3) * np.sin(q4)
    - cof47 * np.cos(q3) * np.sin(q2) * np.sin(q4) * np.sin(q5)
    - cof72 * np.cos(q3) * np.sin(q2) * np.sin(q4) * np.sin(q5)
    - cof47 * np.cos(q2) * np.sin(q3) * np.sin(q4) * np.sin(q5)
    - cof72 * np.cos(q2) * np.sin(q3) * np.sin(q4) * np.sin(q5)
    + cof64 * np.cos(q3) * np.cos(q4) * np.sin(q2) * np.sin(q6)
    + cof64 * np.cos(q2) * np.cos(q4) * np.sin(q3) * np.sin(q6)
    - cof68 * np.cos(q3) * np.cos(q5) * np.sin(q2) * np.sin(q4) * np.sin(q6)
    - cof68 * np.cos(q2) * np.cos(q5) * np.sin(q3) * np.sin(q4) * np.sin(q6)
)
















    G5 = (
    -cof43 * np.cos(q2) * np.cos(q3) * np.cos(q5)
    - cof64 * np.cos(q2) * np.cos(q3) * np.cos(q5) * np.cos(q6)
    + cof47 * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.sin(q2)
    + cof72 * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.sin(q2)
    + cof47 * np.cos(q2) * np.cos(q4) * np.cos(q5) * np.sin(q3)
    + cof72 * np.cos(q2) * np.cos(q4) * np.cos(q5) * np.sin(q3)
    + cof43 * np.cos(q5) * np.sin(q2) * np.sin(q3)
    + cof64 * np.cos(q5) * np.cos(q6) * np.sin(q2) * np.sin(q3)
    + cof47 * np.cos(q2) * np.cos(q3) * np.sin(q5)
    + cof72 * np.cos(q2) * np.cos(q3) * np.sin(q5)
    + cof43 * np.cos(q3) * np.cos(q4) * np.sin(q2) * np.sin(q5)
    + cof64 * np.cos(q3) * np.cos(q4) * np.cos(q6) * np.sin(q2) * np.sin(q5)
    + cof43 * np.cos(q2) * np.cos(q4) * np.sin(q3) * np.sin(q5)
    + cof64 * np.cos(q2) * np.cos(q4) * np.cos(q6) * np.sin(q3) * np.sin(q5)
    - cof47 * np.sin(q2) * np.sin(q3) * np.sin(q5)
    - cof72 * np.sin(q2) * np.sin(q3) * np.sin(q5)
    + cof68 * np.cos(q2) * np.cos(q3) * np.cos(q5) * np.sin(q6)
    - cof68 * np.cos(q5) * np.sin(q2) * np.sin(q3) * np.sin(q6)
    - cof68 * np.cos(q3) * np.cos(q4) * np.sin(q2) * np.sin(q5) * np.sin(q6)
    - cof68 * np.cos(q2) * np.cos(q4) * np.sin(q3) * np.sin(q5) * np.sin(q6)
)

















    G6 = (
    cof68 * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.cos(q6) * np.sin(q2)
    + cof68 * np.cos(q2) * np.cos(q4) * np.cos(q5) * np.cos(q6) * np.sin(q3)
    + cof64 * np.cos(q3) * np.cos(q6) * np.sin(q2) * np.sin(q4)
    + cof64 * np.cos(q2) * np.cos(q6) * np.sin(q3) * np.sin(q4)
    + cof68 * np.cos(q2) * np.cos(q3) * np.cos(q6) * np.sin(q5)
    - cof68 * np.cos(q6) * np.sin(q2) * np.sin(q3) * np.sin(q5)
    + cof64 * np.cos(q3) * np.cos(q4) * np.cos(q5) * np.sin(q2) * np.sin(q6)
    + cof64 * np.cos(q2) * np.cos(q4) * np.cos(q5) * np.sin(q3) * np.sin(q6)
    - cof68 * np.cos(q3) * np.sin(q2) * np.sin(q4) * np.sin(q6)
    - cof68 * np.cos(q2) * np.sin(q3) * np.sin(q4) * np.sin(q6)
    + cof64 * np.cos(q2) * np.cos(q3) * np.sin(q5) * np.sin(q6)
    - cof64 * np.sin(q2) * np.sin(q3) * np.sin(q5) * np.sin(q6)
)



    G = np.array([G1,G2,G3,G4,G5,G6])

    return G



