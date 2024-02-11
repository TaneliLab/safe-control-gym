import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import math
import matplotlib.pyplot as plt
import os 
import yaml

# load hyperparas from yaml
filepath = os.path.join('.','planner.yaml')
with open(filepath, 'r') as file:
    data = yaml.safe_load(file)
# load hyperparameters from yaml file
local_plan_hyperparas = {k: v for d in data['localplan'] for k, v in d.items()}
VERBOSE_PLOT = local_plan_hyperparas['VERBOSE_PLOT']
VMAX = local_plan_hyperparas['VMAX']
AMAX = local_plan_hyperparas['AMAX'] 
LAMBDA_GATES = local_plan_hyperparas['LAMBDA_GATES']
LAMBDA_DRONE = local_plan_hyperparas['LAMBDA_DRONE']
LAMBDA_V = local_plan_hyperparas['LAMBDA_V']
LAMBDA_ACC = local_plan_hyperparas['LAMBDA_ACC']
LAMBDA_OBST = local_plan_hyperparas['LAMBDA_OBST']  # 1500 before
LAMBDA_HEADING = local_plan_hyperparas['LAMBDA_HEADING']

class OnlineLocalReplanner:

    def __init__(self, info_local):

        # sampleRate: to get
        self.global_spline = info_local["global_trajectory"]
        self.spline = info_local["trajectory"]
        self.coeffs = self.spline.c
        self.knot = self.spline.t
        self.t = self.knot[-1]  # current global spline
        self.degree = 5
        self.current_gateID = info_local["current_gate_id"]
        self.current_gate_pos = info_local["current_gate_pose_true"]
        self.sampleRate = info_local["sampleRate"]
        self.obstacle = info_local["nominal_obstacles"]
        # optimize pipeline
        # self.x = spline.c
        # self.num_of_control_points = self.x.shape[0]

        self.x = self.coeffs.flatten()
        # self.len_control_coeffs = len(self.x)
        # # TODO: consider time? or just position
        self.gate_min_dist_knots = info_local["gate_min_dist_knots"]

        self.coeffs_id_gate = self.gateID2controlPoint()
        # self.knot_id_gate = self.gateID2knot()

        self.x = self.x[(self.coeffs_id_gate - 1) *
                        3:(self.coeffs_id_gate + 2) * 3]

        self.current_gate_pos = info_local["current_gate_pose_true"]
        obs = info_local["current_drone_state"]
        self.current_drone_pos = [obs[0], obs[2], obs[4]]
        self.current_time = info_local["current_flight_time"]
        self.drone_obs_stack = info_local["drone_obs_stack"]
        self.gate_pos_stack = info_local["gate_pose_stack"]
        self.current_flight_time_stack = info_local["current_flight_time_stack"]
        self.last_verbose = info_local["last_verbose"]
        self.vmax = VMAX
        self.amax = AMAX
        print("self.knot:", self.knot)
    # def hardGateSwitch(self):
    #     if self.current_gateID >= 0:
    #         coeffs_id = self.gateID2controlPoint()
    #         self.coeffs[coeffs_id] = self.current_gate_pos[0:3]
    #         spline = interpol.BSpline(self.knot, self.coeffs, self.degree)
    #         return spline
    #     else:
    #         return False
        
    def gateID2controlPoint(self):
        if self.current_gateID >= 0:
            coeffs_id = 2 + (self.current_gateID +
                             1) * self.sampleRate  # coeffs_row_id of gate
        else:
            coeffs_id = 0
        return coeffs_id

    def optimizer(self):
        # coeffs_id_gate = self.gateID2controlPoint()
        # TODO: locate idx of coeffs: three key control points around the gate
        # locate idx of knots: three key
        # local_idx = [coeffs_id_gate-1, coeffs_id_gate, coeffs_id_gate+1]
        # # coeffs way
        # mask = np.zeros(self.num_of_control_points)
        # mask[coeffs_id_gate - 1:coeffs_id_gate + 2] = 1

        # flatten way
        # mask = np.zeros(self.len_control_coeffs)
        # mask[(coeffs_id_gate - 1) * 3:(coeffs_id_gate + 2) * 3] = 1

        # self.valid_coeffs_mask = mask

        res = opt.minimize(self.objective,
                           self.x,
                           method='SLSQP',
                           jac=self.numeric_jacobian,
                           tol=1e-2)
        print("finshed opt")
        coeffs_opt = self.unpackx(res.x)
        knots_opt = self.knot
        self.opt_spline = interpol.BSpline(knots_opt, coeffs_opt, self.degree)

        if VERBOSE_PLOT:
            self.plot_xyz_check()
            if self.last_verbose:
                self.plot_xyz()
            # self.plot()
        return self.opt_spline

    def unpackx(self, x):
        coeffs = copy.copy(self.coeffs)

        x_temp = x.reshape(3, -1)  # flatten to array
        # substite three control points: one before gate, one gate, one after gate
        coeffs[(self.coeffs_id_gate - 1):(self.coeffs_id_gate + 2)] = x_temp[:]
        # print("self.coeffs_id_gate :", self.coeffs_id_gate)
        # print("x:", x_temp)
        # print("self.coeffs:", self.coeffs)
        # print("coeffs:", coeffs)
        if math.isnan(x_temp[0][0]):
            assert False
        return coeffs

    def objective(self, x):

        cost = 0

        cost = self.getCost(x)

        return cost

    def numeric_jacobian(self, x):
        # flatten way
        jacobian = []
        dt = 0.01
        for i in range(x.shape[0]):
            new_x = copy.copy(x)
            new_x[i] += dt
            grad = (self.getCost(new_x) - self.getCost(x)) / dt
            jacobian.append(grad)

        return jacobian

    def getCost(self, x):
        cost = 0

        coeffs = self.unpackx(x)
        knots = self.knot
        spline = interpol.BSpline(knots, coeffs, self.degree)

        cost += LAMBDA_HEADING * self.headingCost_local(x, spline)
        cost += LAMBDA_GATES * self.gatesCost_local(x, spline)
        # cost += LAMBDA_V * self.velocityLimitCost(x, spline)
        cost += LAMBDA_ACC * self.accelerationLimitCost(x, spline)
        cost += LAMBDA_DRONE * self.droneCost(x, spline)
        cost += LAMBDA_OBST * self.obstacleCost_strict(x, spline)
        return cost


    def headingCost_local(self, x, spline):
        # only for single gate point
        cost = 0
        dt = 0.1  # smaller more accuarate
        gate_knot = self.gate_min_dist_knots[self.current_gateID]

        positions = spline(gate_knot)  # positions of control points
        before_gate_pos = spline(gate_knot - dt)  # :np.array
        after_gate_pos = spline(gate_knot + dt)
        d = after_gate_pos - before_gate_pos
        # print("gate_knot:", gate_knot)
        # print("d:", d)
        P0 = self.current_gate_pos[0:3]
        N = np.array([
            -np.sin(self.current_gate_pos[5]),
            np.cos(self.current_gate_pos[5]), 0
        ])
        heading_angle_rad = np.arccos(
            np.dot(d, N) / (np.linalg.norm(d) * np.linalg.norm(N) + 0.01))

        heading_angle_deg = abs(np.degrees(heading_angle_rad))
        cost = heading_angle_deg
        # print("current_gate:", self.current_gate_pos)
        # print("heading_angle:", heading_angle_deg)
        return cost

    def gatesCost_local(self, x, spline):

        cost = 0
        # coeffs = self.unpackx(x)
        gate_knot = self.gate_min_dist_knots[self.current_gateID]

        dt = 0.1   # smaller more accurate
        local_gate_knot = np.linspace(gate_knot - dt, gate_knot + dt, 10)
        positions = spline(local_gate_knot)
        # Iterate through waypoints
        P0 = self.current_gate_pos[0:3]
        delta = np.linalg.norm(positions - P0, axis=1)*10  # little trick times 10 to amplify
        cost = np.min(delta)**2

        return cost

    def droneCost(self, x, spline):
        position = spline(self.current_time)
        pos_drone = np.array(self.current_drone_pos)
        delta = np.linalg.norm(position - pos_drone)
        cost = np.min(delta)**2
        return cost

    def obstacleCost_strict(self, x, spline):
        """Penalty for trajectories that are close to obstacles

        Args:
            x (array): opt vector

        Returns:
            cost (scalar): Obstacle penalty
        """
        threshold = 0.5  # penalty on spline points smaller than threshold
        gate_knot = self.gate_min_dist_knots[self.current_gateID]
        dt = 1  # larger to expand control region
        local_gate_knot = np.linspace(gate_knot - dt, gate_knot + dt, 10)
        positions = spline(local_gate_knot)

        cost = 0

        # Iterate through obstacles
        for obst in self.obstacle:
            # print("positions[3]:", positions[:, 2])
            # print("positions[:2]:", positions[:, :2])
            obst_pos = [obst[0], obst[1], 1.05  ]  # TODO: 1.05 is nominal height of obstacle

            # Compute distance between obstacle position and control point
            dist = positions[:, :2] - obst_pos[:2]
            # Norm of the distance
            dist = np.linalg.norm(dist, axis=1)

            delta_height = positions[:, 2] - obst_pos[
                2]  # how much higher than obstacle

            # Select the ones below the threshold(dangerous)
            mask_dist_unsafe = dist < threshold
            mask_height_unsafe = delta_height < 0.1
            mask = [
                a and b for a, b in zip(mask_dist_unsafe, mask_height_unsafe)
            ]
            breached = dist[mask]
            # print("breached:", breached)
            # Cost as the difference between the threshold values and the summed breach of constraint
            cost += (threshold * len(breached) - np.sum(breached))**2

        # 
        
        # also keep the start knot
        return cost

    def velocityLimitCost(self, x, spline):
        # Get control points of velocity spline
        vals = spline.derivative(1).c

        # COmpute the squared norms
        norms = np.linalg.norm(vals, axis=1)

        # Obtain the ones which exceed the limit
        mask = norms > self.vmax

        # Get cost
        cost = np.sum(norms[mask] - self.vmax)**2

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
        vals = spline.derivative(2)
        gate_knot = self.gate_min_dist_knots[self.current_gateID]
        dt = 1  # larger to expand control region
        local_gate_knot = np.linspace(gate_knot - dt, gate_knot + dt, 10)
        velos = vals(local_gate_knot)
        # COmpute the squared norms
        # norms = np.square(np.linalg.norm(vals, axis=1))
        norms = np.linalg.norm(velos, axis=1)
        # Obtain the ones which exceed the limit
        mask = norms > self.amax
        # Get cost
        cost = np.sum(norms[mask] - self.amax)**2

        return cost

    def plot_xyz(self):
        _, axs = plt.subplots(3, 1, figsize=(15, 8))
        time = self.t * np.linspace(0, 1, 100)
        coeffs = self.opt_spline.c
        knots = self.opt_spline.t
        x_coeffs = coeffs[:, 0]
        y_coeffs = coeffs[:, 1]
        z_coeffs = coeffs[:, 2]

        drone_x = self.drone_obs_stack[:, 0]
        drone_y = self.drone_obs_stack[:, 2]
        drone_z = self.drone_obs_stack[:, 4]

        true_gate_x = self.gate_pos_stack[:,0]
        true_gate_y = self.gate_pos_stack[:,1]
        true_gate_z = self.gate_pos_stack[:,2]

        p = self.opt_spline(time)
        p_init = self.global_spline(time)
        axs[0].plot(time, p.T[0], label='local_plan_x')
        axs[0].plot(time, p_init.T[0], label='global_plan_x')
        #  axs[0].scatter(self.opt_spline.t[3:-3], x_coeffs, label='control_x')
        axs[0].scatter(self.gate_min_dist_knots,
                       true_gate_x,marker='o',s=70,
                       label='gate')
        axs[0].scatter(self.current_flight_time_stack,
                       drone_x, marker='*',s=70,
                       label='drone')
        axs[0].legend()
        axs[1].plot(time, p.T[1], label='local_plan_y')
        axs[1].plot(time, p_init.T[1], label='global_plan_y')
        axs[1].scatter(self.gate_min_dist_knots,
                       true_gate_y,marker='o',s=70,
                       label='gate')
        axs[1].scatter(self.current_flight_time_stack,
                       drone_y, marker='*',s=70,
                       label='drone')
        axs[1].legend()
        axs[2].plot(time, p.T[2], label='local_plan_z')
        axs[2].plot(time, p_init.T[2], label='global_plan_z')
        axs[2].scatter(self.gate_min_dist_knots,
                       true_gate_z,marker='o',s=70,
                       label='gate')
        axs[2].scatter(self.current_flight_time_stack,
                       drone_z, marker='*',s=70,
                       label='drone')
        axs[2].legend()
        plt.savefig("./online_plan_data/global_vs_local_xyz.png")
        plt.show()

    def plot_xyz_check(self):
        _, axs = plt.subplots(3, 1)
        time = self.t * np.linspace(0, 1, 100)
        coeffs = self.opt_spline.c
        knots = self.opt_spline.t
        x_coeffs = coeffs[:, 0]
        y_coeffs = coeffs[:, 1]
        z_coeffs = coeffs[:, 2]

        p = self.opt_spline(time)
        p_init = self.global_spline(time)
        axs[0].plot(time, p.T[0], label='opt_x')
        axs[0].plot(time, p_init.T[0], label='init_x')
        #  axs[0].scatter(self.opt_spline.t[3:-3], x_coeffs, label='control_x')
        axs[0].scatter(self.gate_min_dist_knots[self.current_gateID],
                       self.current_gate_pos[0],marker='o',s=20,
                       label='gate')
        axs[0].scatter(self.current_time,
                       self.current_drone_pos[0], marker='*',s=20,
                       label='drone')
        axs[0].legend()
        axs[1].plot(time, p.T[1], label='opt_y')
        axs[1].plot(time, p_init.T[1], label='init_y')
        axs[1].scatter(self.gate_min_dist_knots[self.current_gateID],
                       self.current_gate_pos[1],
                       label='gate')
        axs[1].scatter(self.current_time,
                       self.current_drone_pos[1], marker='*',s=20,
                       label='drone')
        axs[1].legend()
        axs[2].plot(time, p.T[2], label='opt_z')
        axs[2].plot(time, p_init.T[2], label='init_z')
        axs[2].scatter(self.gate_min_dist_knots[self.current_gateID],
                       self.current_gate_pos[2],
                       label='gate')
        axs[2].scatter(self.current_time,
                       self.current_drone_pos[2],
                       label='drone')
        axs[2].legend()
        # plt.savefig("./online_plan_data/global_vs_local_xyz.png")
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def plot(self):
        """Plot the 3d trajectory
        """
        figure = plt.figure(dpi=150)
        ax = figure.add_subplot(projection='3d')
        time = self.t * np.linspace(0, 1, 100)
        p = self.opt_spline(time)
        p_init = self.global_spline(time)

        ax.grid(False)
        ax.plot(p_init.T[0], p_init.T[1], p_init.T[2], label='Init_Traj')
        ax.plot(p.T[0], p.T[1], p.T[2], label='Opt_Traj')

        ax.plot(self.current_gate_pos[0],
                self.current_gate_pos[1],
                self.current_gate_pos[2],
                'o',
                label='gate')
        ax.legend()
        plt.savefig("./online_plan_data/global_vs_local_3d.png")
        plt.show()

if __name__ == "__main__":

    GATES = [
        # x, y, z, r, p, y, type (0: `tall` obstacle, 1: `low` obstacle)
        [0.5, -2.5, 0, 0, 0, -1.57, 0],
        [2, -1.5, 0, 0, 0, 0, 1],
        [0, 0.2, 0, 0, 0, 1.57, 1],
        [-0.5, 1.5, 0, 0, 0, 0, 0]
    ]

    OBSTACLES = [
        # x, y, z, r, p, y
        [1.5, -2.5, 0, 0, 0, 0],
        [0.5, -1, 0, 0, 0, 0],
        [1.5, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0]
    ]

    X0 = [-0.9, -2.9, 0.03]

    GOAL = [-0.5, 2.9, 0.75]

    initial_obs = np.array([-0.9, 0, -2.9, 0, 0.03,0])

    # initial_info =
