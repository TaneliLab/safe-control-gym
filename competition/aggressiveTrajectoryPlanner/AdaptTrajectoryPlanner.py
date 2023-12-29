import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt

# PARAMETERS TO CONTROL THE OPTIMIZATION RESULT ---> SEE competition/trajectoryPlanner/parameters.csv to see which where used

# Maximum velocity allowed before constraining the optimization --> Avoids the necessity to use the model of the inputs and of the dynamics if it remains low
VMAX = 10

AMAX = 8

# If outputs are needed
VERBOSE = False
"""Weights for optimization: 
    The optimization is carried out through an elastic band approach, only soft constraints are enforced and summed in a multiterm cost formulation"""

# Time optimization weight
LAMBDA_T = 20

# Velocity limit weight
LAMBDA_V = 10

LAMBDA_ACC = 10

# Gate attractor weight
LAMBDA_GATES = 100

# Obstacle repulsor weight
LAMBDA_OBST = 100


class TrajectoryPlanner:
    """Trajectory planner class to run through a series of waypoints, given start and goal positions as well as obtstacles

    Returns:
        a trajectory expressed as a scipy non-uniform B-spline with associated duration.
    """

    def __init__(self, start: np.array, goal: np.array, gates, obstacles):
        """Initialization of the class

        Args:
            start (np.array): array with start position
            goal (np.array): array with goal position
            gates (list or array): container of gates postions and orientations
            obstacles (list or array): container of positions
        """

        # Starting position
        self.start = start

        # End position
        self.goal = goal

        # Gates position
        # TODO: fix because we need to find center point
        # Method: Include more control points

        self.gates = gates

        # Waypoints of the trajectory
        self.waypoints = self.setWaypoints()

        # Obstacle positions
        self.obstacles = np.array(obstacles)

        # Time duration of the spline in seconds
        self.t = 6
        self.init_t = self.t

        # B-Spline parametrizing the state-space of the trajectory
        self.spline = self.interpolate()
        self.init_spline = self.spline

        # B-spline coefficients
        self.coeffs = self.spline.c
        """ Optimization variables to pass to the optimization: 
        In this specific case it is necessary to remark that the B-spline is clamped at the beginning and end, in order to include the equality constraints of start and goal positions. 
        Which means that not all control points of the spline are used as optimization parameters, rather we use all of them except the first and last. 
        
        Notably, we could clamp the first and last degree+1 control points, in order to fix the derivatives, however we want to maximize the flexibility of the solution so all higher derivatives can be optimized to improve the performance of the drone"""
        # self.optLim = 3

        # self.optVars = self.coeffs[self.optLim:-self.optLim]

        # # Initial guess of the optmization vector passed to the scipy optimizer:
        # # The vector is composed by the coefficients plus the duration.
        # # The coefficients are a matrix composed of 3-d vectors which are the control points. To use it in the optimization, we flatten them
        # self.x0 = np.append(self.optVars.flatten(), self.t)

        # # Optimization vector used in the clas
        # self.x = self.x0

        # # Velocity constraint enforced
        # self.vmax = VMAX

        # self.amax = AMAX

        # # Cost used during the optimization, initialized here for the gradient computations
        # self.cost = self.getCost(self.x)

    def setWaypoints(self):
        """Sets the waypoints from the gates and start and goal positions"""

        # TODO: select center of gates

        ways = []
        ways.append(self.start)
        for g in self.gates:

            ways.append(g[0:3])

        ways.append(self.goal)
        return np.array(ways)

    def interpolate(self):
        """Interpolate based on waypoints on a fictitious knot vector

        Returns:
            scipy.interolate.Bspline: returns an initial guess for te optimizer, which is the bspline trajectory
        """

        # Compute the initial knot vector to perform interpolation
        self.n = len(self.waypoints)

        knots = np.linspace(0, self.t, self.n)

        # Normalize the knots for later scaling with time
        self.normalized_knots = np.linspace(0, 1, self.n)

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
        self.sampleRate = 2
        keytimesteps = np.linspace(0, self.t, self.n * self.sampleRate)
        self.controlPoints = self.spline(keytimesteps)

        self.spline = interpol.make_interp_spline(keytimesteps,
                                                  self.controlPoints,
                                                  k=self.degree,
                                                  bc_type=bc)
        # exctract the knot vector
        self.knots = self.spline.t
        print("knots:", self.knots)
        # Compute the lenght of the path, it will be used when doing the time optimization
        self.pathLength = self.spline.integrate(self.knots[0], self.knots[-1])

        # Normalization of the knot vector
        self.normalized_knots = self.knots / self.knots[-1]

        # Store the coefficients of the spline
        self.coeffs = self.spline.c
        print("coeffs:", self.coeffs)

        return self.spline

    def plot(self):
        """Plot the 3d trajectory
        """

        ax = plt.figure().add_subplot(projection='3d')

        test = self.t * np.linspace(0, 1, 100)

        p = self.spline(test)

        ax.text(self.start[0], self.start[1], self.start[2], 'Start')
        ax.text(self.goal[0], self.goal[1], self.goal[2], 'Goal')
        ax.set_zlim([0, 2])

        ax.grid(False)
        ax.plot(p.T[0], p.T[1], p.T[2], label='Trajectory')
        # ax.plot(self.waypoints.T[0],
        #         self.waypoints.T[1],
        #         self.waypoints.T[2],
        #         'o',
        #         label='Waypoints')
        ax.plot(self.controlPoints.T[0],
                self.controlPoints.T[1],
                self.controlPoints.T[2],
                'o',
                label='controlPoints')
        ax.legend()
        plt.show()
