import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

VERBOSE = False
VERBOSE_PLOT = False

################ Hard scenrios hypers Start###################
# T = 14  SampleRate=3
# VMAX = 6
# AMAX = 4
# LAMBDA_T = 10
# LAMBDA_GATES = 3000
# LAMBDA_V = 100
# LAMBDA_ACC = 1000
# LAMBDA_OBST = 2000  # 1500 before
# # LAMBDA_TURN = 0
# # LAMBDA_TURN_ANGLE = 0
# LAMBDA_HEADING = 1500
# LAMBDA_INTERSECT = 3000
# LAMBDA_GATEOBST = 1500
# GATE_DT = 0.25

############# Hard scenrios hypers End################

############# Standard Hyper Start ##################
# T = 12 , SampleRate = 3
VMAX = 6
AMAX = 3  # 4 tends to be risky in level3 scenrios: or constraints easy to violate
LAMBDA_T = 10
LAMBDA_GATES = 3000
LAMBDA_V = 100
LAMBDA_ACC = 1000
LAMBDA_OBST = 3000  # 1500 before
# LAMBDA_TURN = 0
# LAMBDA_TURN_ANGLE = 0
LAMBDA_HEADING = 500
LAMBDA_INTERSECT = 3000
LAMBDA_GATEOBST = 1500
GATE_DT = 0.1
############# Standard Hyper End ##################

# Gates properties: {'tall': {'shape': 'square', 'height': 1.0, 'edge': 0.45}, 'low': {'shape': 'square', 'height': 0.525, 'edge': 0.45}}
# Obstacles properties: {'shape': 'cylinder', 'height': 1.05, 'radius': 0.05}
try:
    from aggressiveTrajectoryPlanner.SplineFactory import TrajectoryGenerator
except ImportError:
    from SplineFactory import TrajectoryGenerator


class Globalplanner:

    def __init__(self, spline, initial_obs, initial_info, sampleRate):

        self.sampleRate = sampleRate
        self.obstacle_height = initial_info['obstacle_dimensions'][
            "height"]  #  height 1.05
        self.obstacle_radius = initial_info['obstacle_dimensions'][
            "radius"]  # radius 0.05
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

        self.initial_obs = initial_obs
        self.initial_info = initial_info
        self.tall_gate_height = initial_info["gate_dimensions"]["tall"]["height"]
        self.low_gate_height = initial_info["gate_dimensions"]["low"]["height"]

        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        self.start = (initial_obs[0], initial_obs[2], self.tall_gate_height)
        self.goal = (initial_info["x_reference"][0],
                     initial_info["x_reference"][2],
                     initial_info["x_reference"][4])

        self.waypoints = self.setWaypoints()  # contain start goal
        self.tv = len(self.waypoints)
        # Obstacle positions
        # self.obstacles = np.array(obstacles)

        self.degree = 5
        self.optLim = 3
        self.optVars = self.x[self.optLim:-self.optLim]

        self.gate_min_dist_knots = np.zeros(len(self.NOMINAL_GATES))

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
        for g in self.NOMINAL_GATES:
            height = self.initial_info["gate_dimensions"]["tall"][
                "height"] if g[6] == 0 else self.initial_info[
                    "gate_dimensions"]["low"]["height"]
            ways.append([g[0], g[1], height])

        ways.append(self.goal)
        return np.array(ways)

    def validate(self):
        valid_coeffs_mask = []
        valid_coeffs_index = []
        option = self.valid_mask
        # allow all coeffs except start and goal coeffs
        # To exclude the start(first 9 instead of 3) and goal coeffs
        if option == "ALL":
            for index in range(self.len_control_coeffs +
                               self.len_deltatT_coeffs):
                if (index >= 9 and index < self.len_control_coeffs -
                        9) or index >= self.len_control_coeffs:
                    valid_coeffs_mask.append(1)
                    valid_coeffs_index.append(index)
                else:
                    valid_coeffs_mask.append(0)

        # only allow time coeffs
        elif option == "ONLYTIME":
            for index in range(self.len_control_coeffs +
                               self.len_deltatT_coeffs):
                if index >= self.len_control_coeffs:
                    valid_coeffs_mask.append(1)
                    valid_coeffs_index.append(index)
                else:
                    valid_coeffs_mask.append(0)

        elif option == "ONLYPOS":
            for index in range(self.len_control_coeffs +
                               self.len_deltatT_coeffs):
                if index >= 9 and index < self.len_control_coeffs - 9:
                    valid_coeffs_mask.append(1)
                    valid_coeffs_index.append(index)
                else:
                    valid_coeffs_mask.append(0)

        # can add more
        return valid_coeffs_mask

    def unpackX2deltaT(self, x):
        coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        time_coeffs = x[self.len_control_coeffs:]
        # map to 0.8~1.2
        time_scaling = list(
            map(lambda x: 0.4 / (1 + np.exp(-x)) + 0.8, time_coeffs))
        # map to 0.7~1.3
        time_scaling = list(
            map(lambda x: 0.6 / (1 + np.exp(-x)) + 0.7, time_coeffs))
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
        positions = spline(key_knot)  # positions of control points
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

    def gatesCost_strict(self, x, spline):

        cost = 0
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knot = knots[5:-5]
        t_T = key_knot[-1]
        # print("t_T_gate:", t_T)
        dense_knot = np.linspace(0, t_T, 100)
        positions = spline(dense_knot)

        # Iterate through waypoints
        for idx, w in enumerate(self.waypoints[1:-1]):
            # Compute the distance between the waypoint and the positions
            delta = np.linalg.norm(positions - w, axis=1) * 10  #small trick
            # Select the closest waypoint and penalize the distance
            min_index = np.argmin(delta)
            min_knot = dense_knot[min_index]

            # record knot of passing the gate
            # TODO: potential risk here: don't arrange order of passing gates
            # There will be cases that passing gate2 gate1 gate4 gate3
            self.gate_min_dist_knots[idx] = min_knot
            delta = delta[min_index]
            cost += np.min(delta)**2

        return cost

    def headingCost(self, x, spline):
        # only for single gate point
        cost = 0
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knot = knots[5:-5]

        # only keep knot of the gate control points
        # gate_knot = key_knot[self.sampleRate:-1:self.sampleRate]
        gate_knot = self.gate_min_dist_knots
        #  print("heading Cost key_knot:", key_knot)
        positions = spline(gate_knot)  # positions of control points

        for idx, g in enumerate(self.NOMINAL_GATES):
            # dt = 0.8

            num_samples = 10
            dt_set = np.linspace(0.02, GATE_DT, num_samples)
            for dt in dt_set:
                # idx match to gate knot so we know position to gate
                before_gate_pos = spline(gate_knot[idx] - dt)  # :np.array
                after_gate_pos = spline(gate_knot[idx] + dt)
                d = after_gate_pos - before_gate_pos
                N = np.array([-np.sin(g[5]), np.cos(g[5]), 0])

                heading_angle_rad = np.arccos(
                    np.dot(d, N) / (np.linalg.norm(d) * np.linalg.norm(N)))
                heading_angle_deg = abs(np.degrees(heading_angle_rad))

                cost += heading_angle_deg / num_samples

        return cost

    def intersectCost(self, x, spline):
        # only for single gate point
        cost = 0
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knot = knots[5:-5]

        # only keep knot of the gate control points
        gate_knot = key_knot[self.sampleRate:-1:self.sampleRate]
        gate_knot = self.gate_min_dist_knots
        # print("init gate_knot:", gate_knot)
        #  print("heading Cost key_knot:", key_knot)
        positions = spline(gate_knot)  # positions of control points

        for idx, g in enumerate(self.NOMINAL_GATES):
            # dt = 0.8

            # TODO: make two scheme for simulation and hardware
            height = self.initial_info["gate_dimensions"]["tall"][
                "height"] if g[6] == 0 else self.initial_info[
                    "gate_dimensions"]["low"]["height"]

            # idx match to gate knot so we know position to gate
            num_samples = 10
            dt_set = np.linspace(0.02, GATE_DT, num_samples)
            for dt in dt_set:
                before_gate_pos = spline(gate_knot[idx] - dt)  # :np.array
                after_gate_pos = spline(gate_knot[idx] + dt)
                d = after_gate_pos - before_gate_pos

                P0 = np.array([g[0], g[1], height])
                N = np.array([-np.sin(g[5]), np.cos(g[5]), 0])
                inter = np.dot(N, P0 - before_gate_pos) / np.dot(N, d)

                intersection = before_gate_pos + inter * d

                distance = np.linalg.norm(intersection - P0, axis=0) * 10

                cost += distance**2 / num_samples

        return cost

    def obstacleCost(self, x, spline):
        """Penalty for trajectories that are close to obstacles

        Args:
            x (array): opt vector

        Returns:
            cost (scalar): Obstacle penalty
        """

        threshold = 1  # penalty on control points smaller than threshold
        # coeffs = np.reshape(x[:-1], (-1, 3))
        # coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        coeffs, deltaT = self.unpackX2deltaT(x)
        cost = 0

        # Iterate through obstacles
        for obst in self.NOMINAL_OBSTACLES:

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

    def obstacleCost_strict(self, x, spline):
        """Penalty for trajectories that are close to obstacles

        Args:
            x (array): opt vector

        Returns:
            cost (scalar): Obstacle penalty
        """

        threshold = 0.5  # penalty on spline points smaller than threshold
        # coeffs = np.reshape(x[:-1], (-1, 3))
        # coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knot = knots[5:-5]
        t_T = key_knot[-1]
        dense_knot = np.linspace(0, t_T, 150)
        positions = spline(dense_knot)

        cost = 0
        cost_temp = []
        if len(self.NOMINAL_OBSTACLES) == 0:
            return 0
        # Iterate through obstacles
        for obst in self.NOMINAL_OBSTACLES:
            # print("positions[3]:", positions[:, 2])
            # print("positions[:2]:", positions[:, :2])
            obst_pos = [
                obst[0], obst[1],
                self.initial_info['obstacle_dimensions']["height"]
            ]

            # Compute distance between obstacle position and control point
            dist = positions[:, :2] - obst_pos[:2]
            # Norm of the distance
            dist = np.linalg.norm(dist, axis=1)

            delta_height = positions[:, 2] - obst_pos[
                2]  # how much higher than obstacle
            # print("delta_height：", delta_height)
            # Select the ones below the threshold(dangerous)
            mask_dist_unsafe = dist < threshold
            mask_height_unsafe = delta_height < 0.1
            mask = [
                a and b for a, b in zip(mask_dist_unsafe, mask_height_unsafe)
            ]
            breached = dist[mask]
            # print("breached:", breached)
            # Cost as the difference between the threshold values and the summed breach of constraint
            cost_temp.append((threshold * len(breached) - np.sum(breached))**2)
        cost = max(cost_temp)
        #

        # also keep the start knot

        if VERBOSE:
            print("obstacle cost: ", cost)
        return cost

    def gate_obstacleCost(self, x, spline):

        cost = 0
        threshold = 0.2  # penalty on spline points smaller than threshold
        # coeffs = np.reshape(x[:-1], (-1, 3))
        # coeffs = np.reshape(x[0:self.len_control_coeffs], (-1, 3))
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knot = knots[5:-5]
        t_T = key_knot[-1]
        dense_knot = np.linspace(0, t_T, 200)

        # dense_knot
        risky_hit_gate_knots = copy.copy(dense_knot)
        # print("flex gate knot:", self.gate_min_dist_knots)
        for idx in range(len(self.NOMINAL_GATES)):  # 0 1 2 3
            start_t = self.gate_min_dist_knots[idx] - GATE_DT - 0.1
            end_t = self.gate_min_dist_knots[idx] + GATE_DT + 0.1
            # assert start_t > end_t, "[error]obstacleCost_strict: dt for gate collision avoidance set to high!!"
            risky_hit_gate_knots = risky_hit_gate_knots[
                (risky_hit_gate_knots < start_t) |
                (risky_hit_gate_knots > end_t)]
            assert len(
                risky_hit_gate_knots), "risky_hit_gate_knots has no member!"
        # print("risky_hit_gate_knots:", risky_hit_gate_knots)
        positions_risky = spline(risky_hit_gate_knots)

        self.positions_risky_init = self.init_spline(risky_hit_gate_knots)
        self.risky_hit_gate_knots = risky_hit_gate_knots
        self.positions_risky = positions_risky
        threshold = self.initial_info['gate_dimensions']['tall'][
            'edge'] / 2 + 0.1

        for idx, g in enumerate(self.NOMINAL_GATES):

            # TODO: positions exclude the time when passing, they are kept safe by heading cost

            gate_height = self.initial_info["gate_dimensions"]["tall"][
                "height"] if g[6] == 0 else self.initial_info[
                    "gate_dimensions"]["low"]["height"]

            obst_gate = [g[0], g[1], gate_height]
            dist = np.linalg.norm(positions_risky[:, :2] - obst_gate[:2],
                                  axis=1)
            delta_height = positions_risky[:, 2] - obst_gate[2]
            mask_dist_unsafe = dist < threshold
            mask_hight_unsafe = delta_height < self.initial_info[
                'gate_dimensions']['tall']['edge'] / 2 + 0.05

            mask = [
                a and b for a, b in zip(mask_dist_unsafe, mask_hight_unsafe)
            ]

            breached = dist[mask]
            # print("breached:", breached)
            # Cost as the difference between the threshold values and the summed breach of constraint
            cost += (threshold * len(breached) - np.sum(breached))**2

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

    def velocityLimitCost_strict(self, x, spline):
        cost = 0
        coeffs, deltaT = self.unpackX2deltaT(x)
        knots = self.deltaT2knot(deltaT)
        key_knot = knots[5:-5]
        t_T = key_knot[-1]

        dense_knot = np.linspace(0, t_T, 100)

        val_spline = spline.derivative(1)
        vals = val_spline(dense_knot)
        norms = np.linalg.norm(vals, axis=1)

        # Obtain the ones which exceed the limit
        mask = norms > self.vmax

        # Get cost
        cost = np.sum(norms[mask] - self.vmax)**2

        if VERBOSE:
            # print("mask:", mask)
            # print("vals:", vals)
            print("Velocity limit cost= ", cost)

        return cost

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
        # norms = np.square(np.linalg.norm(vals, axis=1))
        norms = np.linalg.norm(vals, axis=1)
        # Obtain the ones which exceed the limit
        # mask = norms > self.amax**2
        mask = norms > self.amax
        # Get cost
        # cost = np.sum(norms[mask] - self.amax**2)**2
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

        # Constraint Cost

        cost += LAMBDA_GATES * self.gatesCost_strict(x, spline)
        cost += LAMBDA_V * self.velocityLimitCost(x, spline)
        # cost += LAMBDA_V*self.velocityLimitCost_strict(x,spline)
        cost += LAMBDA_ACC * self.accelerationLimitCost(x, spline)
        cost += LAMBDA_OBST * self.obstacleCost_strict(x, spline)
        cost += LAMBDA_HEADING * self.headingCost(x, spline)
        cost += LAMBDA_INTERSECT * self.intersectCost(x, spline)
        cost += LAMBDA_GATEOBST * self.gate_obstacleCost(x, spline)
        # Performance Cost
        cost += LAMBDA_T * self.TimeCost(x, spline)

        # cost += LAMBDA_GATES*self.gatesCost(x, spline)
        # cost += LAMBDA_OBST*self.obstacleCost(x,spline)
        # cost += LAMBDA_TURN * self.TurningCost(x, spline)
        # cost += LAMBDA_TURN_ANGLE * self.TurningCost_OnlyAngle(x, spline)

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

        lower_bounds = np.concatenate(
            [lower_bounds_pos, lower_bound_time_sigmoid])
        upper_bounds = np.concatenate(
            [upper_bounds_pos, upper_bound_time_sigmoid])
        bounds = opt.Bounds(lower_bounds, upper_bounds)
        return bounds

    def optimizer(self):

        #####################Pos Optimize Start#############################
        ###################################################################
        # optimize over control points
        self.valid_mask = "ONLYPOS"
        self.valid_coeffs_mask = self.validate()
        res = opt.minimize(
            self.objective,
            self.x,
            #    method='SLSQP', # try different method
            method='SLSQP',
            jac=self.numeric_jacobian,
            tol=1e-5)  #1e-10

        # self.x = res.x
        x = res.x
        coeffs_opt, deltaT = self.unpackX2deltaT(x)
        self.deltaT = deltaT
        knots_opt = self.deltaT2knot(deltaT)
        self.opt_spline = interpol.BSpline(knots_opt, coeffs_opt, self.degree)
        self.t = knots_opt[-1]
        vals = self.opt_spline.derivative(1).c
        acc = self.opt_spline.derivative(2).c
        print("spline_velo:", np.linalg.norm(vals, axis=1))
        print("spline_acc:", np.linalg.norm(acc, axis=1))

        if VERBOSE_PLOT:
            self.plot_xyz()
            self.plot()
        #####################Pos Optimize End#############################
        ###################################################################

        #####################Time Optimize Start#############################
        ###################################################################

        # optimize over time
        self.valid_mask = "ONLYTIME"
        self.valid_coeffs_mask = self.validate()

        # Thinking of regularization
        res = opt.minimize(
            self.objective,
            self.x,
            # method='trust-constr', # try different method
            method='SLSQP',
            jac=self.numeric_jacobian,
            tol=1e-10)
        self.x = res.x
        x = self.x
        # separate control points and knots
        # copy format from getCost
        coeffs_opt, deltaT = self.unpackX2deltaT(x)
        self.deltaT = deltaT
        knots_opt = self.deltaT2knot(deltaT)
        print("Only time:coeffs_opt", coeffs_opt)
        print("Only time:knots_opt", knots_opt)

        self.opt_spline = interpol.BSpline(knots_opt, coeffs_opt, self.degree)
        self.t = knots_opt[-1]
        vals = self.opt_spline.derivative(1).c
        acc = self.opt_spline.derivative(2).c

        if VERBOSE_PLOT:
            self.plot_xyz()
            self.plot()

        #####################Time Optimize End#############################
        ###################################################################

        vals = self.opt_spline.derivative(1).c
        acc = self.opt_spline.derivative(2).c

        print("spline_velo:", np.linalg.norm(vals, axis=1))
        print("spline_acc:", np.linalg.norm(acc, axis=1))
        print("knots_opt:", knots_opt)
        print("init_coeffs:", self.coeffs0)
        print("final_coeffs:", coeffs_opt)
        print("deltaT0:", self.deltaT0)
        print("deltaT_final:", self.deltaT)
        # copy optimized results

        self.t = knots_opt[-1]
        self.knots = knots_opt
        self.coeffs = coeffs_opt
        self.spline = self.opt_spline

    def get_gate8(self):
        edge = self.initial_info['gate_dimensions']['tall']['edge']/2
        Vertices = []
        Sides = []
        for idx, g in enumerate(self.NOMINAL_GATES):

            gate_height = self.initial_info["gate_dimensions"]["tall"][
                "height"] if g[6] == 0 else self.initial_info[
                    "gate_dimensions"]["low"]["height"]
            N = np.array([-np.sin(g[5]), np.cos(g[5]), 0])
            N_ = np.array([np.cos(g[5]), np.sin(g[5]), 0])

            d_h = [edge*N_[0], edge*N_[1]]
            d_v = [0.05*N[0], 0.05*N[1]]
            center = [g[0], g[1], gate_height]
            vertices = np.array([
                        [center[0]-d_h[0]-d_v[0], center[1]-d_h[1]-d_v[1], gate_height-edge],  # Point 0
                        [center[0]+d_h[0]-d_v[0], center[1]+d_h[1]-d_v[1], gate_height-edge],  # Point 1
                        [center[0]+d_h[0]+d_v[0], center[1]+d_h[1]+d_v[1], gate_height-edge],  # Point 2
                        [center[0]-d_h[0]+d_v[0], center[1]-d_h[1]+d_v[1], gate_height-edge],  # Point 3
                        
                        [center[0]-d_h[0]-d_v[0], center[1]-d_h[1]-d_v[1], gate_height+edge],  # Point 4
                        [center[0]+d_h[0]-d_v[0], center[1]+d_h[1]-d_v[1], gate_height+edge],  # Point 5
                        [center[0]+d_h[0]+d_v[0], center[1]+d_h[1]+d_v[1], gate_height+edge],  # Point 6
                        [center[0]-d_h[0]+d_v[0], center[1]-d_h[1]+d_v[1], gate_height+edge]   # Point 7
                    ])

            sides = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # Side 1
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side 2
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side 3
                    [vertices[2], vertices[3], vertices[7], vertices[6]]   # Side 4
                    ]
            Vertices.append(vertices)
            Sides.append(sides)
        return Sides

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

        coeffs = self.opt_spline.c
        x_coeffs = coeffs[:, 0]
        y_coeffs = coeffs[:, 1]
        z_coeffs = coeffs[:, 2]
        print("t:", self.opt_spline.t)
        print("pos:", x_coeffs)

        axs[0].plot(time, p.T[0], label='opt_spline')
        axs[0].plot(init_time, p_init.T[0], label='init_spline')
        axs[0].scatter(self.opt_spline.t[3:-3],
                       x_coeffs,
                       label='control_opt_x')
        axs[0].legend()
        axs[1].plot(time, p.T[1], label='opt_y')
        axs[1].plot(init_time, p_init.T[1], label='init_y')
        axs[1].scatter(self.opt_spline.t[3:-3],
                       y_coeffs,
                       label='control_opt_y')
        axs[1].legend()
        axs[2].plot(time, p.T[2], label='opt_z')
        axs[2].plot(init_time, p_init.T[2], label='init_z')
        axs[2].scatter(self.opt_spline.t[3:-3],
                       z_coeffs,
                       label='control_opt_z')
        axs[2].legend()
        plt.savefig("./plan_data/global_xyz_plan.jpg")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    def plot(self):
        """Plot the 3d trajectory
        """
        coeffs = self.opt_spline.c
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(projection='3d')

        test = self.t * np.linspace(0, 1, 100)
        init_time = self.init_t * np.linspace(0, 1, 100)

        p = self.opt_spline(test)
        p_init = self.init_spline(init_time)

        ax.text(self.start[0], self.start[1], self.start[2], 'Start')
        ax.text(self.goal[0], self.goal[1], self.goal[2], 'Goal')
        ax.set_zlim([0, 2])

        ax.grid(False)
        # ax.plot(p_init.T[0], p_init.T[1], p_init.T[2], label='Init_Traj')
        ax.plot(p.T[0], p.T[1], p.T[2], label='Opt_Traj')

        # ax.plot(coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], '*', label='control')
        ax.scatter(self.positions_risky.T[0],
                   self.positions_risky.T[1],
                   self.positions_risky.T[2],
                   label='Risky_areas')
        # ax.scatter(self.positions_risky_init.T[0], self.positions_risky_init.T[1], self.positions_risky_init.T[2], label='risky_init')

        ax.plot(self.waypoints.T[0],
                self.waypoints.T[1],
                self.waypoints.T[2],
                'o',
                label='Gates')

        for obst in self.NOMINAL_OBSTACLES:
            obst_x = obst[0]
            obst_y = obst[1]
            obst_z = self.obstacle_height
            radius = self.obstacle_radius
            theta = np.linspace(0, 2 * np.pi, 100)
            z = np.linspace(0, obst_z, 100)
            theta, z = np.meshgrid(theta, z)
            x = obst_x + radius * np.cos(theta)
            y = obst_y + radius * np.sin(theta)
            ax.plot_surface(x, y, z, alpha=0.5, color='b')

        Sides = self.get_gate8()
        for idx, g in enumerate(self.NOMINAL_GATES):
            # Plot the sides
            # Plot each side
            sides = Sides[idx]
            for s in sides:
                ax.add_collection3d(Poly3DCollection([s], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.set_aspect('auto')
        ax.legend()
        plt.savefig("./plan_data/global_3D_plan.jpg")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    # def TurningCost_OnlyAngle(self, x, spline):
    #     cost = 0
    #     # Get coeffs
    #     dt = 0.01
    #     coeffs, deltaT = self.unpackX2deltaT(x)
    #     knots = self.deltaT2knot(deltaT)
    #     key_knots = knots[5:-5]  # only middle time of control points
    #     positions = spline(key_knots)
    #     positions_prime = spline(key_knots + dt)
    #     # knots = self.knots[5:-5]

    #     ## Method 1
    #     # Select only waypoint velo
    #     velos = (positions_prime - positions) / dt
    #     # for loop of all two waypoints
    #     # calculate turning angle(theta) between two waypoints
    #     # get planned time(delta_t) at two waypoints
    #     # get theta/delta_t
    #     for i in range(len(velos) - 1):
    #         dir_1 = velos[i]
    #         dir_2 = velos[i + 1]
    #         cosine_12 = np.dot(
    #             dir_1, dir_2) / (np.linalg.norm(dir_1) * np.linalg.norm(dir_2))
    #         angle_in_rads = np.arccos(cosine_12)
    #         # print(angle_in_rads)
    #         # print(np.degrees(angle_in_rads))
    #         cost += angle_in_rads

    #     if VERBOSE:
    #         print("Turning cost only angle: ", cost)
    #     return cost

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

if __name__ == "__main__":

    # GATES = [[0.5, -2.5, 1.], [2., -1.5, 0.525], [0., 0.2, 0.525],
    #          [-0.5, 1.5, 1.]
    #          #  [-0.9,  1.0  ,  2.   ]
    #          ]
    # # real GATES:
    # GATES = [[0.47, -0.99, 0.52], [-0.5, 0.03, 1.14], [-0.5, 1.02, 0.57],
    #          [0.52, 2.11, 1.15]]

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

    # OBSTACLES = [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0],
    #              [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]
    # OBSTACLES = []
    X0 = [-0.9, -2.9, 0.03]

    GOAL = [-0.5, 2.9, 0.75]

    trajGen = TrajectoryGenerator(X0, GOAL, GATES, OBSTACLES, 3)
    traj = trajGen.spline

    trajReplanar = Globalplanner(traj, X0, GOAL, GATES, OBSTACLES)
    trajReplanar.optimizer()
    # trajReplanar.plot_xyz()
    # trajReplanar.plot()
