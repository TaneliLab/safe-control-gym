

import numpy as np 

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib

class TrajectoryPlanner: 
    """ Trajectory planner class to run through a series of waypoints"""

    def __init__(self, start, goal, gates, obstacles): 

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

        self.t = 1


        self.spline  = self.interpolate()
        
        self.coeffs = self.spline.c

        self.x0 = np.append(self.coeffs.flatten(), self.t)


        self.x = self.x0

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

        self.degree = 5

        #  the number of waypoints nt has to respect the conditions: nt - degree == len(bc_0) + len(bc_r) 
        bc_0 = [(1, np.zeros((3,))), (2, np.zeros((3,)))]
        bc_r = [(1, np.zeros((3,))), (2, np.zeros((3,)))]

        bc = (bc_0, bc_r)

        self.spline = interpol.make_interp_spline(knots, self.waypoints, k = self.degree, bc_type = bc)

        self.knots = self.spline.t

        self.coeffs = self.spline.c
        return self.spline 


    def getCost(self, x): 


        knots = x[-1] * self.knots

        coeffs = np.reshape(self.x[:-1], (-1, 3))
        
        spline = interpol.BSpline(knots, coeffs, self.degree)

        vals = spline.derivative(1).c

        temp = np.linalg.norm(vals, axis=1)

        # cost = np.sum(temp)

        cost = self.obstacleCost(x)


        

        return cost


    def objective(self, x): 

        self.x = x

        self.t = self.x[-1]

        self.coeffs = np.reshape(self.x[:-1], (-1, 3))

        self.cost = self.getCost(x)

        # self.knots = np.linspace(0, self.t, self.n)

        # self.coeffs = self.x[0]

        # self.spline = interpol.BSpline(self.knots, self.coeffs, self.degree)

        # velocity_spline = self.spline.derivative(1)

        # vel_coeffs = velocity_spline.c

        

        return self.cost


    def jacobian(self, x): 

        
        coeffs = x[:-1]
        t = x[-1]

        dt = 1

        jacobian = []
        for i in range(coeffs.shape[0]): 

            coeffs[i] += dt
            new_x = np.append(coeffs, t)
            grad = self.getCost(new_x) - self.cost

            grad /= dt
            jacobian.append(grad)

            coeffs[i] -= dt

        t += dt 

        new_x = (coeffs, t)
        grad = self.getCost(new_x) - self.cost
        grad /= dt
        jacobian.append(grad)

        t -= dt


        return jacobian




    def obstacleCost(self, x): 

        threshold = 0.2
        coeffs = np.reshape(x[:-1], (-1, 3))

        cost = 0
        for obst in self.obstacles: 
            dist = coeffs - obst[:3]

            dist = np.linalg.norm(dist, axis=1)

            mask = dist > threshold

            cost += np.sum(dist[mask])

        print(cost)
        return cost


        


    def optimizer(self): 
        res = opt.minimize(self.objective, self.x, method = 'SLSQP', jac = self.jacobian)

        return res