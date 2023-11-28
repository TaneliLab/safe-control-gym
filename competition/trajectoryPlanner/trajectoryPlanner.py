

import numpy as np 

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt





# PARAMETERS TO CONTROL THE OPTIMIZATION RESULT ---> SEE competition/trajectoryPlanner/parameters.csv to see which where used

# Maximum velocity allowed before constraining the optimization --> Avoids the necessity to use the model of the inputs and of the dynamics if it remains low
VMAX = 10

# If outputs are needed
VERBOSE = False

"""Weights for optimization: 
    The optimization is carried out through an elastic band approach, only soft constraints are enforced and summed in a multiterm cost formulation"""

# Time optimization weight
LAMBDA_T = 0.001

# Velocity limit weight
LAMBDA_V = 10

# Gate attractor weight
LAMBDA_GATES = 100

# Obstacle repulsor weight
LAMBDA_OBST = 1



class TrajectoryPlanner: 
    """Trajectory planner class to run through a series of waypoints, given start and goal positions as well as obtstacles

    Returns:
        a trajectory expressed as a scipy non-uniform B-spline with associated duration.
    """    
    

    def __init__(self, start:np.array, goal:np.array, gates, obstacles): 
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
        self.gates = gates

        # Waypoints of the trajectory
        self.waypoints = self.setWaypoints()


        # Obstacle positions
        self.obstacles = np.array(obstacles)

        # Time duration of the spline in seconds
        self.t = 6

        # B-Spline parametrizing the state-space of the trajectory
        self.spline  = self.interpolate()
        
        # B-spline coefficients
        self.coeffs = self.spline.c

        self.optVars = self.coeffs[1:-1]

        self.x0 = np.append(self.optVars.flatten(), self.t)


        self.x = self.x0


        self.vmax = VMAX

        self.cost = self.getCost(self.x)


        

        

        
        

        
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

        self.n = len(self.waypoints)

        knots = np.linspace(0, self.t, self.n)

        self.normalized_knots = np.linspace(0, 1, self.n)
        
        

        self.degree = 5

        #  the number of waypoints nt has to respect the conditions: nt - degree == len(bc_0) + len(bc_r) 
        bc_0 = [(1, np.zeros((3,))), (2, np.zeros((3,)))]
        bc_r = [(1, np.zeros((3,))), (2, np.zeros((3,)))]

        bc = (bc_0, bc_r)

        self.spline = interpol.make_interp_spline(knots, self.waypoints, k = self.degree, bc_type = bc)



        self.knots = self.spline.t

        self.pathLength = self.spline.integrate(self.knots[0], self.knots[-1])

        self.normalized_knots = self.knots / self.knots[-1]

        self.coeffs = self.spline.c
        return self.spline 
    
    def unpackOptVars(self, x): 

        knots = abs(x[-1]) * self.normalized_knots

        optVars = np.reshape(x[:-1], (-1, 3))



        coeffs = copy.copy(self.coeffs)

        coeffs[1:-1] = optVars


        return knots, coeffs


    def getCost(self, x): 


        knots, coeffs = self.unpackOptVars(x)

        
        spline = interpol.BSpline(knots, coeffs, self.degree)

        cost = 0

        cost +=  LAMBDA_T*self.timeCost(x, spline)

        cost += LAMBDA_OBST * self.obstacleCost(x)

        cost += LAMBDA_GATES*self.gatesCost(x, spline)

        cost += LAMBDA_V * self.velocityLimitCost(x, spline)


        # cost += self.accelerationCost(x)

        # cost += self.velocityCost(x)

        # cost += self.progressCost(x)

        

        return cost


    def objective(self, x): 

        self.x = x

        self.t = abs(self.x[-1])

        knots, coeffs = self.unpackOptVars(x)


        self.coeffs = coeffs

        self.cost = self.getCost(x)

               

        return self.cost


    def jacobian(self, x): 

        
        optVars = x[:-1]
        t = abs(x[-1])

        dt = 1

        jacobian = []
        for i in range(optVars.shape[0]): 

            optVars[i] += dt
            new_x = np.append(optVars, t)
            grad = self.getCost(new_x) - self.cost

            grad /= dt
            jacobian.append(grad)

            optVars[i] -= dt

        t += dt 

        new_x = (optVars, t)
        grad = self.getCost(new_x) - self.cost
        grad /= dt
        jacobian.append(grad)

        t -= dt

        return jacobian
    

    def timeCost(self, x, spline):

        knots, coeffs =  self.unpackOptVars(x)

        current_time = x[-1]
        pathlength = np.linalg.norm(spline.integrate(0, current_time))

        goal_time = pathlength/self.vmax

        goal_time += 0.5*goal_time

        cost = 0

        cost += (current_time - goal_time)**2

        if VERBOSE:
            print('Time cost= ', cost)

        return cost





    def velocityCost(self, x): 

        vmax = 4

        threshold = 0.5 * vmax
        
        knots, coeffs =  self.unpackOptVars(x)
        
        spline = interpol.BSpline(knots, coeffs, self.degree)

        vals = spline.derivative(1).c

        vels = np.linalg.norm(vals, axis = 1)

        delta_v = np.abs(vels - vmax)

        mask = delta_v > threshold

        breached = np.square(delta_v[mask])

        cost = np.sum(breached) 
        

        if VERBOSE: 
          print("Velocity cost: ", cost)
        return cost
    

    def velocityLimitCost(self, x, spline): 

        # knots, coeffs =  self.unpackOptVars(x)


        vals = spline.derivative(1).c

        norms = np.square(np.linalg.norm(vals, axis=1))

        mask = norms > self.vmax**2

        cost = np.sum(norms[mask] - self.vmax**2)**2
        
        if VERBOSE:

            print("Velocity limit cost= ", cost)

        return cost

    def accelerationCost(self, x): 

        amax = 5

        threshold = 0.5 * amax
        
        knots, coeffs =  self.unpackOptVars(x)
        
        spline = interpol.BSpline(knots, coeffs, self.degree)

        vals = spline.derivative(2).c

        acc = np.linalg.norm(vals, axis = 1)

        delta_a = np.abs(acc - amax)

        mask = delta_a > threshold

        breached = np.square(delta_a[mask])

        cost = np.sum(breached) 
        

        if VERBOSE:
            print("Acceleration cost: ", cost)
        return cost

    

    def obstacleCost(self, x): 

        threshold = 0.5
        coeffs = np.reshape(x[:-1], (-1, 3))

        cost = 0
        for obst in self.obstacles: 
            dist = coeffs - obst[:3]

            dist = np.linalg.norm(dist, axis=1)

            mask = dist < threshold

            breached = dist[mask]

            cost +=   threshold * len(breached) - np.sum(breached) 

        if VERBOSE: 
            print("obstacle cost: ", cost)
        return cost


        
    def gatesCost(self, x, spline): 
        threshold = 1

        knots, coeffs = self.unpackOptVars(x)

        cost = 0

        positions = spline(self.knots[5:-5])
        
        for w in self.waypoints: 
            delta = np.linalg.norm(positions - w, axis=1)
            cost += np.min(delta)**2


        # dist = np.abs(positions - self.waypoints)

        # dist = np.linalg.norm(dist, axis=1)
        # mask = dist > threshold
        # breached = dist[mask]
        # cost +=   np.sum(np.square(breached)) 

        # for way in self.waypoints: 
        #     dist = coeffs - way
        #     dist = np.linalg.norm(dist, axis=1)
        #     mask = dist > threshold
        #     breached = dist[mask]
        #     cost +=   np.sum(breached) 

        # print("Gates cost: ", cost)
        return cost

    def progressCost(self, x): 

        knots, coeffs = self.unpackOptVars(x)

        cost = 0

        if VERBOSE: 
            print('Progress Cost: ', cost)

        return cost
        
    def optimizer(self): 

        if VERBOSE: 
            print('Initial coefficients\n', self.coeffs)

        
        print("Starting to plan")

        res = opt.minimize(self.objective, self.x, method = 'SLSQP', jac = self.jacobian)

        
        print("Completed plan")

        self.x = res.x

        self.t = abs(self.x[-1])

        knots, coeffs = self.unpackOptVars(self.x)

        self.knots = knots


        self.coeffs = coeffs

        # if VERBOSE: print('Final coefficients\n', self.coeffs)
        # if VERBOSE: print('Final time: ', self.t)
        
        self.spline = interpol.BSpline(self.knots, self.coeffs, self.degree)

        return res
    

    def plot(self): 

        ax = plt.figure().add_subplot(projection='3d')

        test = self.t * np.linspace(0, 1, 100)


        p = self.spline(test)


        ax.plot(p.T[0], p.T[1], p.T[2])
        ax.plot(self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2],'o')
        plt.show()