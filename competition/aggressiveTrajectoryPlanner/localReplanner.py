import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt
VERBOSE = False
from SplineFactory import TrajectoryGenerator
class LocalReplanner:

    def __init__(self, spline, start: np.array, goal: np.array, gates, obstacles):

        self.init_spline = spline
        self.spline = spline
        self.coeffs0 = spline.c
        self.knot0 = spline.t
        self.t = self.knot0[-1]
        self.init_t = self.t
        # self.x = self.coeffs0.flatten()
        self.x = np.append(self.coeffs0.flatten(), self.knot0[5:-5])

        self.knots = self.knot0
        self.coeffs = self.coeffs0

        self.start = start
        self.goal = goal
        self.gates = gates
        self.waypoints = self.setWaypoints()
        self.tv = len(self.waypoints)

        self.degree = 5
        self.optLim = 3
        self.optVars = self.x[self.optLim:-self.optLim]
        self.valid_coeffs_mask = self.validate()

        
        print("self.coeffs0", self.coeffs0.shape)
        print("velo:", spline.derivative(1).c.shape)
        print("velo:", spline.derivative(1).c)
        print("acc:", spline.derivative(2).c.shape)
        print("knots:", self.knots.shape)
        print("knots:", self.knots)
        print("x:",self.x)

    def setWaypoints(self):
        """Sets the waypoints from the gates and start and goal positions"""

        # TODO: select center of gates

        ways = []
        ways.append(self.start)
        for g in self.gates:

            ways.append(g[0:3])

        ways.append(self.goal)
        return np.array(ways)

    def validate(self):
        # self.optVars = self.x[self.optLim:-self.optLim]
        self.n = self.x.shape[0]
        # valid_coeffs_mask = [0 for i in range(self.n + self.tv)]
        # valid_coeffs_index = [i for i in range(self.n)] # take 0~n-1 
        # valid_coeffs_index = valid_coeffs_index[3:-3] # take 3~n-3
        # for i in range(self.n, self.n*2):
        #     valid_coeffs_index.append(i)
        # # valid_coeffs_index = [4,5,6]
        
        # for index in valid_coeffs_index:
        #     if index<self.n:
        #         valid_coeffs_mask[index] = 1
        valid_coeffs_mask = []
        valid_coeffs_index = []
        for index in range(self.n + self.tv):
            if (index>=3 and index<self.n-3) or index>=self.n+1: # starttime=0
                valid_coeffs_mask.append(1)
                valid_coeffs_index.append(index)
            else:
                valid_coeffs_mask.append(0)

        print("valid_coeffs_index: ", valid_coeffs_index)
        if VERBOSE:
            print("valid_coeffs_index: ", valid_coeffs_index)
        return valid_coeffs_mask


    def objective(self, x):
        self.x = x
        self.cost = self.getCost(x)

        return self.cost

    def getCost(self,x):
        cost = 0
        coeffs = np.reshape(x, (-1, 3))
        spline = interpol.BSpline(self.knots, coeffs, self.degree)
        cost += self.gatesCost(x, spline)
        cost += self.TurningCost(x, spline)

        return cost


    def gatesCost(self, x, spline):
        """Cost value that pushes the spline towards the waypoints in the middle of the gates

        Args:
            x (array): opt vector
            spline (Bspline): current b-spline

        Returns:
            cost (scalar): Gates distance penalty
        """

        cost = 0
        # Compute a number of key positions
        positions = spline(self.knots[5:-5])
        # print("c:",spline.c)

        # Iterate through waypoints
        for w in self.waypoints:
            # Compute the distance between the waypoint and the positions
            delta = np.linalg.norm(positions - w, axis=1)
            # Select the closest waypoint and penalize the distance
            cost += np.min(delta)**2
        if VERBOSE:
            print("Gates cost: ", cost)
        return cost
    
    def TurningCost(self,x,spline):
        cost = 0 
        # Get control points 
        # key_time = self.knots[5:-5]
        dt = 0.05
        positions = spline(self.knots[5:-5])
        positions_prime = spline(self.knots[5:-5]+dt)
        knots = self.knots[5:-5]
        ## Method 1
        # Select only waypoint velo 
        velos = (positions_prime - positions)/dt
        # for loop of all two waypoints
        # calculate turning angle(theta) between two waypoints
        # get planned time(delta_t) at two waypoints
        # get theta/delta_t
        for i in range(len(velos)-1):
            dir_1 = velos[i]
            dir_2 = velos[i+1]
            cosine_12 = np.dot(dir_1, dir_2)/(np.linalg.norm(dir_1)*np.linalg.norm(dir_2))
            angle_in_rads = np.arccos(cosine_12)
            delta_t = knots[i+1] - knots[i]
            # print(angle_in_rads)
            # print(np.degrees(angle_in_rads))
            cost += angle_in_rads/delta_t

        if VERBOSE:
            print("Turning cost: ", cost)
        return cost
    
    def TimeCost(self,x,spline):
        cost = 0

        # To shorten time for all partion between waypoints
        if VERBOSE:
            print("Time cost: ", cost)
        return cost


    def numeric_jacobian(self,x):
        dt = 0.1
        jacobian = []
        for i in range(x.shape[0]):
            
            if self.valid_coeffs_mask[i] == 0:
                jacobian.append(0)
            else:
                new_x = copy.copy(x)
                new_x[i] += dt
                grad = (self.getCost(new_x) - self.getCost(x))/dt
                jacobian.append(grad)

        # TODO: add time jacobian
        if VERBOSE:
            print("jacobian:", jacobian)
        return jacobian



    def optimizer(self):
        res = opt.minimize(self.objective,
                    self.x,
                    method='SLSQP',
                    jac=self.numeric_jacobian,
                    tol=1e-10)
        
        self.x = res.x
        print("x:",self.x)
        self.coeffs = np.reshape(self.x, (-1,3))
        self.spline = interpol.BSpline(self.knots, self.coeffs, self.degree)

    def plot_xyz(self):
        """Plot the xyz trajectory
        """

        _, axs = plt.subplots(3, 1)

        time = self.t * np.linspace(0, 1, 100)
        init_time = self.init_t * np.linspace(0, 1, 100)

        p = self.spline(time)
        p_init = self.init_spline(init_time)

        axs[0].plot(time, p.T[0], label='opt_x')
        axs[0].plot(init_time, p_init.T[0], label='init_x')
        axs[0].legend()
        axs[1].plot(time, p.T[1], label='opt_y')
        axs[1].plot(init_time, p_init.T[1], label='init_y')
        axs[1].legend()
        axs[2].plot(time, p.T[2], label='opt_z')
        axs[2].plot(init_time, p_init.T[2], label='init_z')
        axs[2].legend()
        plt.show()


    def plot(self):
        """Plot the 3d trajectory
        """

        ax = plt.figure().add_subplot(projection='3d')

        test = self.t * np.linspace(0, 1, 100)
        init_time = self.init_t * np.linspace(0, 1, 100)

        p = self.spline(test)
        p_init = self.init_spline(init_time)

        ax.text(self.start[0], self.start[1], self.start[2], 'Start')
        ax.text(self.goal[0], self.goal[1], self.goal[2], 'Goal')
        ax.set_zlim([0, 2])

        ax.grid(False)
        ax.plot(p_init.T[0], p_init.T[1], p_init.T[2], label='Init_Traj')
        ax.plot(p.T[0], p.T[1], p.T[2], label='Opt_Traj')
        
        ax.plot(self.waypoints.T[0],
                self.waypoints.T[1],
                self.waypoints.T[2],
                'o',
                label='Waypoints')
        ax.legend()
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

    trajGen = TrajectoryGenerator(X0,GOAL, GATES,OBSTACLES)
    traj = trajGen.spline

    trajReplanar = LocalReplanner(traj, X0,GOAL, GATES,OBSTACLES)
    trajReplanar.optimizer()
    trajReplanar.plot_xyz()
    trajReplanar.plot()
    

