import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt

INIT_FLIGHT_TIME = 7
class TrajectoryGenerator:

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

        # Time duration of the spline in seconds
        self.t = INIT_FLIGHT_TIME
        self.init_t = self.t

        # B-Spline parametrizing the state-space of the trajectory
        self.spline = self.interpolate()
        self.init_spline = self.spline

        self.x = self.spline.c.flatten()
        self.new_x = self.x
        # self.new_x = self.x + 0.1

    def setWaypoints(self):
        """Sets the waypoints from the gates and start and goal positions"""

        # TODO: select center of gates

        ways = []
        ways.append(self.start)
        for g in self.gates:
            # ways.append(g[0:3])
            ways.append([g[0], g[1], g[2]])

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

        # Interpolation of the same B-spline with more control points
        self.sampleRate = 2
        self.n = self.n * self.sampleRate
        keytimesteps = np.linspace(0, self.t, self.n)
        self.controlPoints = self.spline(keytimesteps)

        self.spline = interpol.make_interp_spline(keytimesteps,
                                                  self.controlPoints,
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