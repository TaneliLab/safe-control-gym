import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt

# INIT_FLIGHT_TIME = 12
class TrajectoryGenerator:

    def __init__(self, initial_obs, initial_info, sampleRate,
                 flight_time_init):
        """Initialization of the class

        Args:
            start (np.array): array with start position
            goal (np.array): array with goal position
            gates (list or array): container of gates postions and orientations
            obstacles (list or array): container of positions
        """
        self.initial_obs = initial_obs
        self.initial_info = initial_info
        HEIGHT_TALL_GATE = initial_info["gate_dimensions"]["tall"]["height"]
        HEIGHT_LOW_GATE = initial_info["gate_dimensions"]["low"]["height"]
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Starting position
        self.start = (initial_obs[0], initial_obs[2], HEIGHT_TALL_GATE)

        # End position
        self.goal = (initial_info["x_reference"][0],
                     initial_info["x_reference"][2],
                     initial_info["x_reference"][4])

        # Gates position
        # TODO: fix because we need to find center point
        # Method: Include more control points

        # self.gates = gates

        # Waypoints of the trajectory
        self.waypoints = self.setWaypoints()
        self.n = len(self.waypoints)  # 6: conatin start goal

        # Time duration of the spline in seconds
        self.t = flight_time_init
        self.init_t = self.t

        # B-Spline parametrizing the state-space of the trajectory
        self.sampleRate = sampleRate

        # self.spline = self.interpolate_twoForGate()  #two gate points mode
        self.spline = self.interpolate_single_gate()  # one gate mode

        self.init_spline = self.spline

        self.x = self.spline.c.flatten()
        self.new_x = self.x

        # show the inital pos velo acc spline
        #  self.plot_spline()


    def setWaypoints(self):
        """Sets the waypoints from the gates and start and goal positions"""

        # TODO: select center of gates

        ways = []
        ways.append(self.start)
        for idx, g in enumerate(self.NOMINAL_GATES):
            height = self.initial_info["gate_dimensions"]["tall"][
                "height"] if g[6] == 0 else self.initial_info[
                    "gate_dimensions"]["low"]["height"]
            ways.append((g[0], g[1], height))

        ways.append(self.goal)
        return np.array(ways)

    def interpolate_twoForGate(self):
        """Interpolate based on waypoints on a fictitious knot vector

        Returns:
            scipy.interolate.Bspline: returns an initial guess for te optimizer, which is the bspline trajectory
        """

        # Compute the initial knot vector to perform interpolation

        knots = np.linspace(0, self.t, self.n)

        # # Normalize the knots for later scaling with time
        # self.normalized_knots = np.linspace(0, 1, self.n)

        # Degree of the curve, selected based on the number of waypoints and the requested smoothness
        self.degree = 5

        # Boundary conditions on higher derivatives, initialized at zero to give a coutios initial guess to the optimizer
        #  the number of waypoints nt has to respect the conditions: nt - degree == len(bc_0) + len(bc_r)
        bc_0 = [(1, np.zeros((3, ))), (2, np.zeros((3, )))]
        bc_r = [(1, np.zeros((3, ))), (2, np.zeros((3, )))]

        bc = (bc_0, bc_r)

        # Interpolation of the B-spline based on scipy rules
        self.spline = interpol.make_interp_spline(knots,
                                                  self.waypoints,
                                                  k=self.degree,
                                                  bc_type=bc)

        # Interpolation of the same B-spline with more control points

        num_control_points = self.n + (self.n - 1) * (self.sampleRate - 1)
        timesteps = np.linspace(0, self.t, num_control_points)
        delta_t = timesteps[1] - timesteps[0]
        controlPoints = self.spline(timesteps)

        gatetimesteps = timesteps[self.sampleRate:-1:self.sampleRate]
        gatePoints = self.spline(gatetimesteps)
        print("timesteps:", timesteps)
        print("controlPoints:", controlPoints)
        factory_timesteps = timesteps[0:self.sampleRate]  # t0 t1 t_(g1-1)
        factory_controlPoints = controlPoints[0:self.sampleRate, :]

        delta = 0.2
        deltat_scale = 0.1
        for idx, g in enumerate(self.NOMINAL_GATES):
            height = self.initial_info["gate_dimensions"]["tall"][
                "height"] if g[6] == 0 else self.initial_info[
                    "gate_dimensions"]["low"]["height"]
            delta_p = [-delta * np.sin(g[5]), delta * np.cos(g[5]), 0]
            gate_idx = (idx + 1) * self.sampleRate

            before_gate_pos = gatePoints[idx, :] - delta_p
            after_gate_pos = gatePoints[idx, :] + delta_p
            two_gate_pos = np.array([before_gate_pos, after_gate_pos])

            before_gate_time = gatetimesteps[idx] - delta_t * deltat_scale
            after_gate_time = gatetimesteps[idx] + delta_t * deltat_scale
            two_gate_time = np.array([before_gate_time, after_gate_time])
            factory_timesteps = np.hstack((factory_timesteps, two_gate_time))
            middle_time = timesteps[(idx + 1) * self.sampleRate + 1:(idx + 2) *
                                    self.sampleRate]
            factory_timesteps = np.hstack((factory_timesteps, middle_time))

            factory_controlPoints = np.vstack(
                (factory_controlPoints, two_gate_pos))
            middle_point = controlPoints[(idx + 1) * self.sampleRate +
                                         1:(idx + 2) * self.sampleRate, :]
            factory_controlPoints = np.vstack(
                (factory_controlPoints, middle_point))

        factory_controlPoints = np.vstack(
            (factory_controlPoints, controlPoints[-1, :]))
        factory_timesteps = np.hstack((factory_timesteps, timesteps[-1]))

        # print("factory_controlPoints:", factory_controlPoints)
        # print("factory_timesteps:", factory_timesteps)
        # print("factory_controlPoints:", factory_controlPoints.shape)
        # print("factory_timesteps:", factory_timesteps.shape)

        self.spline = interpol.make_interp_spline(factory_timesteps,
                                                  factory_controlPoints,
                                                  k=self.degree,
                                                  bc_type=bc)

        # exctract the knot vector
        self.knots = self.spline.t
        print("init knots:", self.knots)
        # Compute the lenght of the path, it will be used when doing the time optimization
        self.pathLength = self.spline.integrate(self.knots[0], self.knots[-1])

        # Normalization of the knot vector
        self.normalized_knots = self.knots / self.knots[-1]

        # Store the coefficients of the spline
        self.coeffs = self.spline.c
        print("init coeffs:", self.coeffs)

        return self.spline

    def interpolate_single_gate(self):
        """Interpolate based on waypoints on a fictitious knot vector

        Returns:
            scipy.interolate.Bspline: returns an initial guess for te optimizer, which is the bspline trajectory
        """

        # Compute the initial knot vector to perform interpolation

        knots = np.linspace(0, self.t, self.n)

        # # Normalize the knots for later scaling with time
        # self.normalized_knots = np.linspace(0, 1, self.n)

        # Degree of the curve, selected based on the number of waypoints and the requested smoothness
        self.degree = 5

        # Boundary conditions on higher derivatives, initialized at zero to give a coutios initial guess to the optimizer
        #  the number of waypoints nt has to respect the conditions: nt - degree == len(bc_0) + len(bc_r)
        bc_0 = [(1, np.zeros((3, ))), (2, np.zeros((3, )))]
        bc_r = [(1, np.zeros((3, ))), (2, np.zeros((3, )))]

        bc = (bc_0, bc_r)

        # Interpolation of the B-spline based on scipy rules
        self.spline = interpol.make_interp_spline(knots,
                                                  self.waypoints,
                                                  k=self.degree,
                                                  bc_type=bc)

        # Interpolation of the same B-spline with more control points

        num_control_points = self.n + (self.n - 1) * (self.sampleRate - 1)
        timesteps = np.linspace(0, self.t, num_control_points)
        delta_t = timesteps[1] - timesteps[0]
        controlPoints = self.spline(timesteps)
        print("timesteps:", timesteps)
        print("controlPoints:", controlPoints)

        self.spline = interpol.make_interp_spline(timesteps,
                                                  controlPoints,
                                                  k=self.degree,
                                                  bc_type=bc)

        # exctract the knot vector
        self.knots = self.spline.t
        print("init knots:", self.knots)
        # Compute the lenght of the path, it will be used when doing the time optimization
        self.pathLength = self.spline.integrate(self.knots[0], self.knots[-1])

        # Normalization of the knot vector
        self.normalized_knots = self.knots / self.knots[-1]

        # Store the coefficients of the spline
        self.coeffs = self.spline.c
        print("init coeffs:", self.coeffs)

        return self.spline

    def plot_spline(self):
        pos_spline = self.spline
        velo_spline = self.spline.derivative(1)
        acc_spline = self.spline.derivative(2)
        knot = self.spline.t

        time = self.t * np.linspace(0, 1, 100)

        pos = pos_spline(time)
        velo = velo_spline(time)
        acc = acc_spline(time)

        # just verify velocity spline is same with velocity_instantous
        # dt = 0.01
        # velo_in = []
        # for t in time[0:-1]:
        #     p_t = pos_spline(t)
        #     p_tprime = pos_spline(t + dt)

        #     v_t = (p_tprime - p_t)/dt
        #     velo_in.append(v_t)

        # velo_in.append([0,0,0])
        # velo_in = np.array(velo_in)
        # print("velo_in:", velo_in)

        _, axs = plt.subplots(3, 1)
        axs[0].plot(time, pos.T[0], label='position_spline')
        axs[0].plot(time, velo.T[0], label='velocity_spline')
        # axs[0].plot(time, velo_in[:,0], label='velocity_instantous')
        axs[0].plot(time, acc.T[0], label='acceleration_spline')
        axs[0].legend()

        axs[1].plot(time, pos.T[1], label='position_spline')
        axs[1].plot(time, velo.T[1], label='velocity_spline')
        axs[1].plot(time, acc.T[1], label='acceleration_spline')
        axs[1].legend()

        axs[2].plot(time, pos.T[2], label='position_spline')
        axs[2].plot(time, velo.T[2], label='velocity_spline')
        axs[2].plot(time, acc.T[2], label='acceleration_spline')
        axs[2].legend()
        plt.show()


if __name__ == "__main__":

    GATES = [[ 0.5,   -2.5  ,  1.   ],
    [ 2. , -1.5  ,  0.525],
    [ 0. ,  0.2  ,  0.525],
    [-0.5,  1.5  ,  1.   ]
    #  [-0.9,  1.0  ,  2.   ]
    ]



    OBSTACLES = [
        [1.5, -2.5, 0, 0, 0, 0],
        [0.5, -1, 0, 0, 0, 0],
        [1.5, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0]
        ]


    X0 = [ -0.9, -2.9,  1]

    GOAL= [-0.5, 2.9, 0.75]

    trajGen = TrajectoryGenerator(X0, GOAL, GATES,OBSTACLES)
    traj = trajGen.spline
    print(traj.c)
    print(traj.t)
    print('x-------------')
    print(trajGen.x)
    print(trajGen.new_x)
