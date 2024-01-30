import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import math
import matplotlib.pyplot as plt

VERBOSE_PLOT = False
VMAX = 6
AMAX = 4
LAMBDA_GATES = 4000
LAMBDA_DRONE = 1000
LAMBDA_V = 0
LAMBDA_ACC = 1000
LAMBDA_HEADING = 1000

# Say as failure case
class OnlineLocalReplanner:

    def __init__(self, spline, sampleRate, current_gateID, current_gate_pos,
                 obs, time):

        # sampleRate: to get
        self.spline = spline
        self.coeffs = spline.c
        self.knot = spline.t
        self.t = self.knot[-1]
        self.degree = 5
        self.current_gateID = current_gateID
        self.current_gate_pos = current_gate_pos
        self.sampleRate = sampleRate

        # optimize pipeline
        # self.x = spline.c
        # self.num_of_control_points = self.x.shape[0]

        self.x = self.coeffs.flatten()
        # self.len_control_coeffs = len(self.x)
        # # TODO: consider time? or just position
        self.coeffs_id_gate = self.gateID2controlPoint()
        self.knot_id_gate = self.gateID2knot()
        self.knot[self.knot_id_gate]

        self.x = self.x[(self.coeffs_id_gate - 1) *
                        3:(self.coeffs_id_gate + 2) * 3]

        self.current_gate_pos = current_gate_pos
        self.current_drone_pos = [obs[0], obs[2], obs[4]]
        self.current_time = time
        self.vmax = VMAX
        self.amax = AMAX
        print("self.knot:", self.knot)

    def gateID2controlPoint(self):
        if self.current_gateID >= 0:
            coeffs_id = 3 + (self.current_gateID +
                             1) * self.sampleRate  # coeffs_row_id of gate
        else:
            coeffs_id = 0
        return coeffs_id

    def gateID2knot(self):
        if self.current_gateID >= 0:
            knots_id = 6 + (self.current_gateID +
                            1) * self.sampleRate  # knot_id of gate
        else:
            knots_id = 0
        return knots_id

    # def hardGateSwitch(self):
    #     if self.current_gateID >= 0:
    #         coeffs_id = self.gateID2controlPoint()
    #         self.coeffs[coeffs_id] = self.current_gate_pos[0:3]
    #         spline = interpol.BSpline(self.knot, self.coeffs, self.degree)
    #         return spline
    #     else:
    #         return False

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
            self.plot_xyz()
            self.plot()
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
        return cost


    def headingCost_local(self, x, spline):
        # only for single gate point
        cost = 0
        dt = 0.1  # smaller more accuarate

        # coeffs = self.unpackx(x)
        # print("headingCost_local_coeffs:", coeffs)

        local_idx = [
            self.coeffs_id_gate - 1, self.coeffs_id_gate,
            self.coeffs_id_gate + 1
        ]

        gate_knot = self.knot[self.knot_id_gate]

        #  print("heading Cost key_knot:", key_knot)
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
        gate_knot = self.knot[self.knot_id_gate]
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
        gate_knot = self.knot[self.knot_id_gate]
        dt = 0.6  # larger to expand control region
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
        _, axs = plt.subplots(3, 1)
        time = self.t * np.linspace(0, 1, 100)
        coeffs = self.opt_spline.c
        knots = self.opt_spline.t
        x_coeffs = coeffs[:, 0]
        y_coeffs = coeffs[:, 1]
        z_coeffs = coeffs[:, 2]

        p = self.opt_spline(time)
        p_init = self.spline(time)
        axs[0].plot(time, p.T[0], label='opt_x')
        axs[0].plot(time, p_init.T[0], label='init_x')
        #  axs[0].scatter(self.opt_spline.t[3:-3], x_coeffs, label='control_x')
        axs[0].scatter(self.opt_spline.t[self.knot_id_gate],
                       self.current_gate_pos[0],
                       label='gate')
        axs[0].scatter(self.current_time,
                       self.current_drone_pos[0],
                       label='drone')
        axs[0].legend()
        axs[1].plot(time, p.T[1], label='opt_y')
        axs[1].plot(time, p_init.T[1], label='init_y')
        axs[1].scatter(self.opt_spline.t[self.knot_id_gate],
                       self.current_gate_pos[1],
                       label='gate')
        axs[1].scatter(self.current_time,
                       self.current_drone_pos[1],
                       label='drone')
        axs[1].legend()
        axs[2].plot(time, p.T[2], label='opt_z')
        axs[2].plot(time, p_init.T[2], label='init_z')
        axs[2].scatter(self.opt_spline.t[self.knot_id_gate],
                       self.current_gate_pos[2],
                       label='gate')
        axs[2].scatter(self.current_time,
                       self.current_drone_pos[2],
                       label='drone')
        axs[2].legend()
        plt.show()

    def plot(self):
        """Plot the 3d trajectory
        """
        ax = plt.figure().add_subplot(projection='3d')
        time = self.t * np.linspace(0, 1, 100)
        p = self.opt_spline(time)
        p_init = self.spline(time)

        ax.grid(False)
        ax.plot(p_init.T[0], p_init.T[1], p_init.T[2], label='Init_Traj')
        ax.plot(p.T[0], p.T[1], p.T[2], label='Opt_Traj')

        ax.plot(self.current_gate_pos[0],
                self.current_gate_pos[1],
                self.current_gate_pos[2],
                'o',
                label='gate')
        ax.legend()
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
