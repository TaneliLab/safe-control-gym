

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
LAMBDA_T = 0.005

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

        
        """ Optimization variables to pass to the optimization: 
        In this specific case it is necessary to remark that the B-spline is clamped at the beginning and end, in order to include the equality constraints of start and goal positions. 
        Which means that not all control points of the spline are used as optimization parameters, rather we use all of them except the first and last. 
        
        Notably, we could clamp the first and last degree+1 control points, in order to fix the derivatives, however we want to maximize the flexibility of the solution so all higher derivatives can be optimized to improve the performance of the drone"""
        self.optVars = self.coeffs[1:-1]

        # Initial guess of the optmization vector passed to the scipy optimizer: 
            # The vector is composed by the coefficients plus the duration. 
            # The coefficients are a matrix composed of 3-d vectors which are the control points. To use it in the optimization, we flatten them
        self.x0 = np.append(self.optVars.flatten(), self.t)

        # Optimization vector used in the clas
        self.x = self.x0

        # Velocity constraint enforced
        self.vmax = VMAX

        # Cost used during the optimization, initialized here for the gradient computations
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
        bc_0 = [(1, np.zeros((3,))), (2, np.zeros((3,)))]
        bc_r = [(1, np.zeros((3,))), (2, np.zeros((3,)))]

        bc = (bc_0, bc_r)

        # Interpolation of the B-spline based on scipy rules
        self.spline = interpol.make_interp_spline(knots, self.waypoints, k = self.degree, bc_type = bc)

        # exctract the knot vector
        self.knots = self.spline.t
        
        # Compute the lenght of the path, it will be used when doing the time optimization
        self.pathLength = self.spline.integrate(self.knots[0], self.knots[-1])

        # Normalization of the knot vector
        self.normalized_knots = self.knots / self.knots[-1]

        # Store the coefficients of the spline
        self.coeffs = self.spline.c


        return self.spline 
    
    def unpackOptVars(self, x): 
        """Unpacks the optimization variable x which is returned by the optimizer, in order to produce a more convenient representation as coefficients and duration

        Args:
            x (array): Optimization vector

        Returns:
            knots (array): new knot vector scaled with new duration
            coeffs (array): matrix of 3-d control points 
        """
        # Scale knot vector for derivatives computation
        knots = abs(x[-1]) * self.normalized_knots

        # Extract the coefficients which were optimized and reshape them
        optVars = np.reshape(x[:-1], (-1, 3))

        # Copied in order not to modify the coefficients stored as class attribute
        coeffs = copy.copy(self.coeffs)

        # Update the coefficients
        coeffs[1:-1] = optVars


        return knots, coeffs


    def getCost(self, x): 
        """Helper function that takes care of the cost computation for the optimizer. It is also used to compute the gradients
        

        Args:
            x (array): optimization vector

        Returns:
            cost (scalar): Numerical value of the cost
        """

        # Extract updated variables
        knots, coeffs = self.unpackOptVars(x)

        # Construct updated spline
        spline = interpol.BSpline(knots, coeffs, self.degree)

        # Initialize cost
        cost = 0

        # Cost as a summation of multiple terms, either for optimization of the trajectory or soft constraints

        # Time optimization
        cost +=  LAMBDA_T*self.timeCost(x, spline)

        # Obstacle avoidance
        cost += LAMBDA_OBST * self.obstacleCost(x)

        # Gates crossing
        cost += LAMBDA_GATES*self.gatesCost(x, spline)

        # Velocity limiting
        cost += LAMBDA_V * self.velocityLimitCost(x, spline)

      

        return cost


    def objective(self, x): 
        """Objective of the optimization, takes care of updating the necessary class attributes so that other functions do not modify or access them and can be used in gradient computation

        Args:
            x (array): Optimizatin vector

        Returns:
            cost (scalar): Cost
        """

        self.x = x

        self.t = abs(self.x[-1])

        knots, coeffs = self.unpackOptVars(x)


        self.coeffs = coeffs

        self.cost = self.getCost(x)

               

        return self.cost


    def jacobian(self, x): 
        """Function to compute the gradients. As it is not trivial, prone to errors and time consuming to analytically compute gradients with respect to b-spline coefficients, we do it numerically

        Args:
            x (array): Optimization vector

        Returns:
            jacobian (list): Jacobian of the cost wrt the optimization variables -> J = dC/dx returns an vector of the size of x
        """

        
        optVars = x[:-1]
        t = abs(x[-1])

        # Perturbation numerical derivative
        dt = 1

        jacobian = []

        # iterate through variables
        for i in range(optVars.shape[0]): 
            
            # increase value
            optVars[i] += dt
            new_x = np.append(optVars, t)

            # compute perturbed cost and subtract from the stored one
            grad = self.getCost(new_x) - self.cost

            # divide by perturbation
            grad /= dt

            # append the numerical gradient
            jacobian.append(grad)

            # restore variable value to avoid issues
            optVars[i] -= dt

        # Perform the same for the time
        t += dt 

        new_x = (optVars, t)
        grad = self.getCost(new_x) - self.cost
        grad /= dt
        jacobian.append(grad)

        t -= dt

        return jacobian
    

    def timeCost(self, x, spline):
        """Time related cost for time-optimal trajectory generation. 
        Promotes fast trajectories that reach the goal. 
        
        It takes in a spline, computes its length and determines the best possible time to traverse it using the VMAX parameter-> goal_time
        The cost is the squared difference between the duration and the goal time, which is discounted by 50% to accomodate safety and feasibility. 

        N.B. This cost term alone leads to a fast trajectories with small duration. It promotes paths that do not traverse the gates.

        Args:
            x (array): opt vector
            spline (Bspline): current b-spline

        Returns:
            cost (scalar): timecost
        """

        knots, coeffs =  self.unpackOptVars(x)

        # Get time
        current_time = x[-1]

        # Compute length
        pathlength = np.linalg.norm(spline.integrate(0, current_time))

        # Obtain and discount goal time
        goal_time = pathlength/self.vmax

        goal_time += 0.5*goal_time

        cost = 0

        # Compute cost
        cost += (current_time - goal_time)**2

        if VERBOSE:
            print('Time cost= ', cost)

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

            print("Velocity limit cost= ", cost)

        return cost

    
    

    def obstacleCost(self, x): 
        """Penalty for trajectories that are close to obstacles

        Args:
            x (array): opt vector

        Returns:
            cost (scalar): Obstacle penalty
        """

        threshold = 0.5
        coeffs = np.reshape(x[:-1], (-1, 3))

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

            # Cost as the difference between the threshold values and the summed breach of constraint
            cost +=   (threshold * len(breached) - np.sum(breached) )**2

        if VERBOSE: 
            print("obstacle cost: ", cost)
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
        
        # Iterate through waypoints
        for w in self.waypoints: 

            # Compute the distance between the waypoint and the positions
            delta = np.linalg.norm(positions - w, axis=1)

            # Select the closest waypoint and penalize the distance
            cost += np.min(delta)**2


        # print("Gates cost: ", cost)
        return cost

        
    def optimizer(self): 
        """Member function that computes the actual optimization. We use SLSQ because it is very reliable and allows to avoid computing the Hessian numerically 

        Returns:
            res (scipy.optimize.OptimizeResult): The result of the optimization
        """

        if VERBOSE: 
            print('Initial coefficients\n', self.coeffs)

        
        print("Starting to plan")

        # Perform the optimization by selecting the objective function, the optimization vector and the jacobian
        res = opt.minimize(self.objective, self.x, method = 'SLSQP', jac = self.jacobian)

        
        print("Completed plan")

        # Extract coefficients and duration from the result

        self.x = res.x

        self.t = abs(self.x[-1])

        knots, coeffs = self.unpackOptVars(self.x)

        # Update class attributes
        self.knots = knots


        self.coeffs = coeffs

        if VERBOSE: 
            print('Final coefficients\n', self.coeffs)
        if VERBOSE: 
            print('Final time: ', self.t)
        
        self.spline = interpol.BSpline(self.knots, self.coeffs, self.degree)

        self.rotationalTrajectory()

        

        return res
    

    def plot(self): 
        """Plot the 3d trajectory
        """

        ax = plt.figure().add_subplot(projection='3d')

        test = self.t * np.linspace(0, 1, 100)


        p = self.spline(test)

        ax.text(self.start[0], self.start[1], self.start[2], 'Start' )
        ax.text(self.goal[0], self.goal[1], self.goal[2], 'Goal' )
        ax.set_zlim([0, 2])

        ax.grid(False)
        ax.plot(p.T[0], p.T[1], p.T[2], label='Trajectory')
        ax.plot(self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2],'o', label='Waypoints')
        ax.legend()
        plt.show()


    def sampleForce(self): 
        
        n = 100

        t_samples = np.linspace(0, self.t, n)

        self.samplingDeltaT = self.t / n

        accelerationSpline = self.spline.derivative(2)

        accelerations = accelerationSpline(t_samples)

        forceDirections = accelerations - 9.81

        norms = np.linalg.norm(forceDirections, axis=1)

        forceDirections = np.divide(forceDirections.T, norms)

        forceDirections = forceDirections.T


        return forceDirections
    
    def sampleOrientations(self, forceDirections): 

        n = np.array([1,0,0])

        z_vecs = forceDirections


        y_vecs = np.cross(z_vecs, n)

        norms = np.linalg.norm(y_vecs, axis=1)

        y_vecs = np.divide(y_vecs.T, norms)

        y_vecs = y_vecs.T

        x_vecs = np.cross(y_vecs, z_vecs)

        matrices = [np.eye(3,3)]

        for i in range(x_vecs.shape[0]): 
            x = x_vecs[i].T
            y = y_vecs[i].T
            z = z_vecs[i].T
            R = [x, y, z]

            matrices.append(np.array(R).T)
            
        matrices.append(matrices[-1])

        return matrices
    
    def differentiateMatrices(self, matrices): 
        
        R_dots = []

        for i in range(len(matrices)-1): 
            m0 = matrices[i]
            m1 = matrices[i+1]

            dm = m1 - m0 

            dm = dm / self.samplingDeltaT 

            R_dots.append(dm)

        return R_dots
    
    def getOmegas(self, Rs, R_dots): 

        omegas = [np.array([0,0,0])]


        for i in range(len(Rs)-1): 
            R_trans = Rs[i].T

            R_dot = R_dots[i]

            omega_hat = R_dot * R_trans

            omega = [omega_hat[2,1], omega_hat[0, 2], omega_hat[1,0]]
            omegas.append(omega)


        return np.array(omegas)
    
    def interpolateOmegas(self, omegas): 

        omega_knots = np.linspace(0, self.t, omegas.shape[0])

        self.omega_spline = interpol.make_interp_spline(omega_knots, omegas)

        


    def rotationalTrajectory(self): 
        forcesDirs = self.sampleForce()

        Rs = self.sampleOrientations(forcesDirs)

        R_dots = self.differentiateMatrices(Rs)

        omegas = self.getOmegas(Rs, R_dots)

        self.interpolateOmegas(omegas)
        
