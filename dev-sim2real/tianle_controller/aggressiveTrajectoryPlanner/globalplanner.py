import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt

VERBOSE = False
VERBOSE_PLOT = False
VMAX = 3
AMAX = 6
LAMBDA_T = 1
LAMBDA_GATES = 100
LAMBDA_V = 100
LAMBDA_ACC = 100
LAMBDA_OBST = 0
LAMBDA_TURN = 0
LAMBDA_TURN_ANGLE = 1

try:    
    from aggressiveTrajectoryPlanner.SplineFactory import TrajectoryGenerator
except ImportError: 
    from SplineFactory import TrajectoryGenerator

class Globalplanner:

    def __init__(self, spline, start: np.array, goal: np.array, gates,
                 obstacles):

        self.init_spline = copy.copy(spline)
        self.spline = copy.copy(spline)
        self.coeffs0 = self.init_spline.c
        self.knot0 = self.init_spline.t
        self.t = self.knot0[-1]
        self.init_t = self.t

        # self.x = self.coeffs0.flatten()

        # include control points and knot time
        self.deltaT0 = self.knot2deltaT(self.knot0)
        self.deltaT = self.knot2deltaT(self.knot0)
        self.time_coeffs = [0 for i in range(len(self.deltaT))]
        # self.x = np.append(self.coeffs0.flatten(), self.deltaT)
        self.x = np.append(self.coeffs0.flatten(), self.time_coeffs)
        

        self.x_init = self.x
        self.len_control_coeffs = len(self.coeffs0.flatten())
        self.len_deltatT_coeffs = len(self.deltaT)

        self.knots = self.knot0
        self.coeffs = self.coeffs0

        self.start = start
        self.goal = goal
        self.gates = gates
        self.waypoints = self.setWaypoints()
        self.tv = len(self.waypoints)
        # Obstacle positions
        self.obstacles = np.array(obstacles)

        self.degree = 5
        self.optLim = 3
        self.optVars = self.x[self.optLim:-self.optLim]

        # Only update selected coefficients
        self.valid_mask = "ONLYTIME"
        self.valid_coeffs_mask = self.validate()
        self.vmax = VMAX
        self.amax = AMAX
        print("self.coeffs0", self.coeffs0.shape)
        print("velo:", spline.derivative(1).c.shape)
        print("velo:", spline.derivative(1).c)
        print("acc:", spline.derivative(2).c.shape)
        print("knots:", self.knots.shape)
        print("knots:", self.knots)
        print("x:", self.x)

    def knot2deltaT(self, knots):
        deltaT = []
        knots = knots[5:-5]
        for i in range(len(knots) - 1):
            t1 = knots[i]
            t2 = knots[i + 1]
            deltaT.append(t2 - t1)
        return deltaT

    def deltaT2knot(self, deltaT):
        # update whole self.knots

        knots = copy.copy(self.knot0)  # it prevent knot0 is changed
        local_knot = [0]
        time = 0
        for deltat in deltaT:
            local_knot.append(time +
                              abs(deltat))  # trick: ensure non-decreasing
            time += abs(deltat)
        for i in range(5):
            local_knot.append(time)
        knots[5:] = local_knot

        # !update tail all to same value
        return knots

    # def initKnotSchedule(self):
    #     pathlength = np.linalg.norm(self.spline.integrate(0, self.init_t))

    #     return 0

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
        valid_coeffs_mask = []
        valid_coeffs_index = []
        option = self.valid_mask 
        # allow all coeffs except start and goal coeffs
        # To exclude the start(first 9 instead of 3) and goal coeffs
        if option == "ALL":
            for index in range(self.len_control_coeffs + self.len_deltatT_coeffs):
                if (index >= 9 and index < self.len_control_coeffs -
                        9) or index >= self.len_control_coeffs:
                    valid_coeffs_mask.append(1)
                    valid_coeffs_index.append(index)
                else:
                    valid_coeffs_mask.append(0)

        # only allow time coeffs 
        elif option =="ONLYTIME":
            for index in range(self.len_control_coeffs + self.len_deltatT_coeffs):
                if index >= self.len_control_coeffs:
                    valid_coeffs_mask.append(1)
                    valid_coeffs_index.append(index)
                else:
                    valid_coeffs_mask.append(0)

        elif option =="ONLYPOS":
            for index in range(self.len_control_coeffs + self.len_deltatT_coeffs):
                if index >= 9 and index < self.len_control_coeffs - 9:
                    valid_coeffs_mask.append(1)
                    valid_coeffs_index.append(index)
                else:
                    valid_coeffs_mask.append(0)
        return valid_coeffs_mask

    def unpackX2deltaT(self,x):
        coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        time_coeffs = x[self.len_control_coeffs:]
        # map to 0.8~1.2
        time_scaling = list(map(lambda x: 0.4/(1+np.exp(-x)) + 0.8, time_coeffs))
        # map to 0.7~1.3
        time_scaling = list(map(lambda x: 0.6/(1+np.exp(-x)) + 0.7, time_coeffs))
        deltaT = self.deltaT0.copy()
        for i in range(len(deltaT)):
            deltaT[i] *= time_scaling[i]
        return coeffs, deltaT  

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
        # deltaT = x[self.len_control_coeffs:]
        coeffs, deltaT = self.unpackX2deltaT(x)

        knots = self.deltaT2knot(deltaT)

        key_knot = knots[5:-5]
        positions = spline(key_knot) # positions of control points
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
    
    def obstacleCost(self, x, spline):
        """Penalty for trajectories that are close to obstacles

        Args:
            x (array): opt vector

        Returns:
            cost (scalar): Obstacle penalty
        """

        threshold = 1 # penalty on control points smaller than threshold
        # coeffs = np.reshape(x[:-1], (-1, 3))
        # coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        coeffs, deltaT = self.unpackX2deltaT(x)
        cost = 0

        # Iterate through obstacles
        for obst in self.obstacles:

            # Compute distance between obstacle position and control point
            dist = coeffs - obst[:3]

            # Norm of the distance
            dist = np.linalg.norm(dist, axis=1)

            # Select the ones below the threshold
            mask = dist < threshold
            breached = dist[mask]
            # print("breached:", breached)
            # Cost as the difference between the threshold values and the summed breach of constraint
            cost += (threshold * len(breached) - np.sum(breached))**2

        if VERBOSE:
            print("obstacle cost: ", cost)
        return cost
    
    def TurningCost_OnlyAngle(self, x, spline):
        cost = 0
        # Get coeffs
        dt = 0.01
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knots = knots[5:-5]  # only middle time of control points
        positions = spline(key_knots)
        positions_prime = spline(key_knots + dt)
        # knots = self.knots[5:-5]

        ## Method 1
        # Select only waypoint velo
        velos = (positions_prime - positions) / dt
        # for loop of all two waypoints
        # calculate turning angle(theta) between two waypoints
        # get planned time(delta_t) at two waypoints
        # get theta/delta_t
        for i in range(len(velos) - 1):
            dir_1 = velos[i]
            dir_2 = velos[i + 1]
            cosine_12 = np.dot(
                dir_1, dir_2) / (np.linalg.norm(dir_1) * np.linalg.norm(dir_2))
            angle_in_rads = np.arccos(cosine_12)
            # print(angle_in_rads)
            # print(np.degrees(angle_in_rads))
            cost += angle_in_rads 

        if VERBOSE:
            print("Turning cost only angle: ", cost)
        return cost

    def TurningCost(self, x, spline):
        cost = 0
        # Get control points
        # key_time = self.knots[5:-5]
        dt = 0.02
        # deltaT = x[self.len_control_coeffs:]
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knots = knots[5:-5]  # only middle time of control points
        positions = spline(key_knots)
        positions_prime = spline(key_knots + dt)
        # knots = self.knots[5:-5]

        ## Method 1
        # Select only waypoint velo
        velos = (positions_prime - positions) / dt
        # for loop of all two waypoints
        # calculate turning angle(theta) between two waypoints
        # get planned time(delta_t) at two waypoints
        # get theta/delta_t
        for i in range(len(velos) - 1):
            dir_1 = velos[i]
            dir_2 = velos[i + 1]
            cosine_12 = np.dot(
                dir_1, dir_2) / (np.linalg.norm(dir_1) * np.linalg.norm(dir_2))
            angle_in_rads = np.arccos(cosine_12)
            delta_t = key_knots[i + 1] - key_knots[i]
            # print(angle_in_rads)
            # print(np.degrees(angle_in_rads))
            cost += angle_in_rads / delta_t

        if VERBOSE:
            print("Turning cost: ", cost)
        return cost

    def TimeCost(self, x, spline):
        cost = 0
        # deltaT = x[self.len_control_coeffs:]
        coeffs, deltaT = self.unpackX2deltaT(x)
        for deltat in deltaT:
            cost += deltat
        # To shorten time for all partion between waypoints
        cost = cost**2
        if VERBOSE:
            print("Time cost: ", cost)
        return cost

    def velocityLimitCost(self, x, spline):
        """Soft constraint on the velocity. Adds a quadratic penaly whenever the norm of the velocity exceeds the VMAX value in the control points. 
            It is conservative as the control points define a convex hull within which the velocity is confined.

        Args:
            x (array): opt vector
            spline (Bspline): current b-spline

        Returns:
            cost (scalar): Velocity penalty
        """

        # Get control points of velocity spline
        vals = spline.derivative(1).c

        # COmpute the squared norms
        norms = np.square(np.linalg.norm(vals, axis=1))
        
        # Obtain the ones which exceed the limit
        mask = norms > self.vmax**2

        # Get cost
        cost = np.sum(norms[mask] - self.vmax**2)**2

        if VERBOSE:
            # print("mask:", mask)
            # print("vals:", vals)
            print("Velocity limit cost= ", cost)

        return cost

    def accelerationLimitCost(self, x, spline):
        """Soft constraint on the velocity. Adds a quadratic penaly whenever the norm of the velocity exceeds the VMAX value in the control points. 
            It is conservative as the control points define a convex hull within which the velocity is confined.

        Args:
            x (array): opt vector
            spline (Bspline): current b-spline

        Returns:
            cost (scalar): Velocity penalty
        """

        # Get control points of velocity spline
        vals = spline.derivative(2).c

        # COmpute the squared norms
        norms = np.square(np.linalg.norm(vals, axis=1))
        norms = np.linalg.norm(vals, axis=1)
        # Obtain the ones which exceed the limit
        mask = norms > self.amax**2
        mask = norms > self.amax
        # Get cost
        cost = np.sum(norms[mask] - self.amax**2)**2
        cost = np.sum(norms[mask] - self.amax)**2
        if VERBOSE:

            print("Acceleration limit cost= ", cost)

        return cost

    def objective(self, x):
        self.x = x
        self.cost = self.getCost(x)

        return self.cost

    def getCost(self, x):

        #! all coeffs and knots should use local variables
        cost = 0
        # take control points only
        # coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        # # update the deltaT everytime getCost from objective and jacobian
        # deltaT = x[self.len_control_coeffs:]
        coeffs, deltaT = self.unpackX2deltaT(x)
        
        knots = self.deltaT2knot(deltaT)
        # update spline for cost
        spline = interpol.BSpline(knots, coeffs, self.degree)
        # print("x:", x)
        # print("getcost_deltaT:", deltaT)

        #TODO:Make it flexible to choose cost by control LAMBDA
        cost += LAMBDA_GATES*self.gatesCost(x, spline)
        cost += LAMBDA_T*self.TimeCost(x,spline)
        cost += LAMBDA_V*self.velocityLimitCost(x,spline)
        cost += LAMBDA_ACC*self.accelerationLimitCost(x,spline)
        cost += LAMBDA_OBST*self.obstacleCost(x,spline)
        cost += LAMBDA_TURN * self.TurningCost(x, spline)
        cost += LAMBDA_TURN_ANGLE * self.TurningCost_OnlyAngle(x, spline)

        # cost += self.KnotCost(x,spline,0.5)

        return cost
    

    def numeric_jacobian(self, x):
        # x 0:self.n-self.tv control points,  self.n-self.tv: time
        dt = 0.01
        lr = 0.01
        jacobian = []
        
        for i in range(x.shape[0]):
            if i < self.len_control_coeffs:
                # self.len_control_coeffs = 30
                if self.valid_coeffs_mask[i] == 0:
                    jacobian.append(0)
                else:
                    new_x = copy.copy(x)
                    new_x[i] += dt
                    # print("new_x:", new_x)
                    # print("x:",x)
                    grad = (self.getCost(new_x) - self.getCost(x)) / dt
                    # print("grad:", grad)
                    jacobian.append(grad)
            else:
                if self.valid_coeffs_mask[i] == 0:
                    jacobian.append(0)
                else:
                    new_x = copy.copy(x)
                    # new_x[i] = new_x[i]*2/(1+np.exp(-dt))
                    new_x[i] += lr
                    grad = (self.getCost(new_x) - self.getCost(x)) / lr
                    jacobian.append(grad)
        # if VERBOSE:
        #     print("jacobian:", jacobian)
        return jacobian

    def bounds(self):
        
        # print(self.waypoints)
        # print(self.coeffs0)
        # print(self.time_coeffs)
        n_pos = len(self.coeffs0)
        n_time = len(self.time_coeffs)
        lb_x, lb_y, lb_z = -3, -3, -0.1
        ub_x, ub_y, ub_z = 3, 3, 2

        lb_x, lb_y, lb_z = -30, -30, -10
        ub_x, ub_y, ub_z = 30, 30, 20

        lower_bounds_pos = np.array([lb_x, lb_y, lb_z] * n_pos)
        upper_bounds_pos = np.array([ub_x, ub_y, ub_z] * n_pos)

        # lower_bound_acc_velo = np.array([-0.1, -0.1, -0.1] * 2)
        # upper_bound_acc_velo = np.array([0.1, 0.1, 0.1] * 2)

        lower_bound_time_sigmoid = np.array([-100] * n_time)
        upper_bound_time_sigmoid = np.array([100] * n_time)

        lower_bounds = np.concatenate([lower_bounds_pos, lower_bound_time_sigmoid])
        upper_bounds = np.concatenate([upper_bounds_pos, upper_bound_time_sigmoid])
        bounds = opt.Bounds(lower_bounds, upper_bounds)
        return bounds

    def optimizer(self):
        # optimize over control points 
        self.valid_mask = "ONLYPOS"
        self.valid_coeffs_mask = self.validate()
        bounds = self.bounds()
        res = opt.minimize(self.objective,
                           self.x,
                        #    method='SLSQP', # try different method
                           method='SLSQP',
                           jac=self.numeric_jacobian,
                           tol=1e-10)

        # self.x = res.x
        x = res.x
        coeffs_opt, deltaT = self.unpackX2deltaT(x)
        self.deltaT = deltaT
        knots_opt = self.deltaT2knot(deltaT)
        self.opt_spline = interpol.BSpline(knots_opt, coeffs_opt, self.degree)
        self.t = knots_opt[-1]
        if VERBOSE_PLOT:
            self.plot_xyz()
            self.plot()


        # optimize over time
        self.valid_mask = "ONLYTIME"
        self.valid_coeffs_mask = self.validate()

        # Thinking of regularization
        res = opt.minimize(self.objective,
                           self.x,
                           method='trust-constr', # try different method
                           jac=self.numeric_jacobian,
                           tol=1e-10)
        self.x = res.x
        x = self.x
        # separate control points and knots
        # copy format from getCost
        coeffs_opt, deltaT = self.unpackX2deltaT(x)
        self.deltaT = deltaT
        knots_opt = self.deltaT2knot(deltaT)
        print("knots_opt:", knots_opt)
        print("init_coeffs:", self.coeffs0)
        print("final_coeffs:", coeffs_opt)
        print("deltaT0:", self.deltaT0)
        print("deltaT_final:", self.deltaT)
        self.opt_spline = interpol.BSpline(knots_opt, coeffs_opt, self.degree)
        vals = self.opt_spline.derivative(1).c
        print("spline_velo:",np.linalg.norm(vals, axis=1) )
        acc = self.opt_spline.derivative(2).c
        print("spline_acc:",np.linalg.norm(acc, axis=1) )

        # copy optimized results
        self.t = knots_opt[-1]
        self.knots = knots_opt
        self.coeffs = coeffs_opt
        self.spline = self.opt_spline
        if VERBOSE_PLOT:
            self.plot_xyz()
            self.plot()


    def plot_xyz(self):
        """Plot the xyz trajectory
        """

        _, axs = plt.subplots(3, 1)
        knot0 = self.deltaT2knot(self.deltaT0)
        knot = self.deltaT2knot(self.deltaT)

        time = self.t * np.linspace(0, 1, 100)
        init_time = self.init_t * np.linspace(0, 1, 100)

        p = self.opt_spline(time)
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

        p = self.opt_spline(test)
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

    GATES = [[0.5, -2.5, 1.], [2., -1.5, 0.525], [0., 0.2, 0.525],
             [-0.5, 1.5, 1.]
             #  [-0.9,  1.0  ,  2.   ]
             ]
    # real GATES:
    GATES = [[0.47, -0.99, 0.52], [-0.5, 0.03, 1.14], [-0.5, 1.02, 0.57],
             [0.52, 2.11, 1.15]]
    
    GATES = [[0.47, -0.99, 0.52, 0, 0, 0.8, 1],
      [-0.5, 0.03, 1.14, 0, 0, 0, 0],
      [-0.5, 1.02, 0.57, 0, 0, 0, 1],
      [0.52, 2.11, 1.15, 0, 0, 0, 0]
    ]
    # 
    OBSTACLES = [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0],
                 [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], 
                 [2, -1.4, 0, 0, 0, 0], [1.8, -1.4, 0, 0, 0, 0],  # extra
                 [2.3, -1.4, 0, 0, 0, 0], [0, 0.23, 0, 0, 0, 0]]  # extra
    
    OBSTACLES = [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0],
                 [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]
    OBSTACLES = []
    X0 = [-0.9, -2.9, 0.03]

    GOAL = [-0.5, 2.9, 0.75]

    trajGen = TrajectoryGenerator(X0, GOAL, GATES, OBSTACLES, 3)
    traj = trajGen.spline

    trajReplanar = Globalplanner(traj, X0, GOAL, GATES, OBSTACLES)
    trajReplanar.optimizer()
    # trajReplanar.plot_xyz()
    # trajReplanar.plot()
