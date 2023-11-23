import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class MinimumSnapTrajectory:
    def __init__(self,params, gates, obstacles):
        # traj_plan_params = {"ctrl_time": initial_info["episode_len_sec"], "ctrl_freq": self.CTRL_FREQ, "gate_sequence_fixed": True,
        #         "start_pos": start_pos, "stop_pos": goal_pos, "max_recursion_num": adjustable_params['max_recursion_num'],
        #         "uav_radius": 0.075, "obstacle_geo": obstacle_geo, "gate_geo": gate_geo, "accuracy": 0.01,
        #         "gate_collide_angle": adjustable_params['gate_collide_angle'], "gate_height": gate_height,
        #         "path_insert_point_dist_min": adjustable_params['path_insert_point_dist_min'],
        #         "gate_waypoint_safe_dist": adjustable_params['gate_waypoint_safe_dist'],
        #         "traj_max_vel": adjustable_params['traj_max_vel'], "traj_gamma": adjustable_params['traj_gamma']}

        print("Initializing trajectory generator...")

        self.obstacle_geo = params["obstacle_geo"]
        self.gate_geo = params["gate_geo"]
        self.gate_height = params["gate_height"]

        gates = np.array(gates)
        for i in range(gates.shape[0]):
            gates[i][2] = self.gate_height[int(gates[i][6])]
        self.gates = gates[0:6] if gates.ndim == 1 else gates[:,0:6]
        self.obstacles = np.array(obstacles)
        self.episode_len_sec = params["ctrl_time"]
        self.ctrl_freq = params["ctrl_freq"]
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.gate_sequence_fixed = params["gate_sequence_fixed"]
        self.start_pos = params["start_pos"]
        self.stop_pos = params["stop_pos"][0:3]
        self.accuracy = params["accuracy"]
        # self.max_recursion_num = int(params["max_recursion_num"])

        self.uav_radius = params["uav_radius"]
        # self.gate_collide_angle = params["gate_collide_angle"]
        # self.gate_waypoint_safe_dist = params["gate_waypoint_safe_dist"]
        # self.path_insert_point_dist_min = params["path_insert_point_dist_min"]

        # self.traj_max_vel = params["traj_max_vel"]
        # self.traj_gamma = params["traj_gamma"]
    
    def objective_function(self, coefficients):
        # Objective function to minimize (snap)
        snap_coefficients = np.polyder(coefficients, 4)
        return np.linalg.norm(snap_coefficients)

    def constraint_position(self, coefficients, times, waypoints):
        # Constraint: Position at waypoints
        return [np.polyval(coefficients, t) - waypoint for t, waypoint in zip(times, waypoints)]

    def constraint_velocity(self, coefficients, times):
        # Constraint: Velocity at specific times (e.g., start and end)
        velocity_coefficients = np.polyder(coefficients, 1)
        return [np.polyval(velocity_coefficients, t) for t in [times[0], times[-1]]]

    def constraint_acceleration(self, coefficients, times):
        # Constraint: Acceleration at specific times (e.g., start and end)
        acceleration_coefficients = np.polyder(coefficients, 2)
        # boarder_constraint = [np.polyval(acceleration_coefficients, t) for t in [self.times[0], self.times[-1]]]
        # maximum_constraint = []
        return [np.polyval(acceleration_coefficients, t) for t in [times[0], times[-1]]]
        
    def constraint_acceleration_max(self,coefficients, times, waypoints, bound):
        acceleration_coefficients = np.polyder(coefficients, 2)
        constant_limit = bound # Adjust this to your desired constant limit
        return [constant_limit - np.abs(np.polyval(acceleration_coefficients, t)) for t in np.linspace(times[0], times[-1], len(waypoints))]

    def constraint_jerk(self, coefficients, times):
        # Constraint: Jerk at specific times (e.g., start and end)
        jerk_coefficients = np.polyder(coefficients, 3)
        return [np.polyval(jerk_coefficients, t) for t in [times[0], times[-1]]]

    # def optimize_trajectory(self):
    #     # Initial guess for polynomial coefficients (random values)
    #     initial_guess = np.random.rand(self.num_coefficients)

    #     # Bounds for the optimization variables
    #     bounds = [(-10, 10) for _ in range(self.num_coefficients)]

    #     # Constraints
    #     constraints = [
    #         {'type': 'eq', 'fun': self.constraint_position},
    #         {'type': 'eq', 'fun': self.constraint_velocity},
    #         {'type': 'eq', 'fun': self.constraint_acceleration},
    #         {'type': 'eq', 'fun': self.constraint_jerk},
    #         {'type': 'ineq', 'fun': self.constraint_acceleration_max}
    #     ]

    #     # Optimization
    #     result = minimize(self.objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    #     print(result)
    #     # Extracting the optimal coefficients
    #     optimal_coefficients = result.x
    #     return optimal_coefficients
    

    # def plot_trajectory_with_velocity_and_acceleration(self, waypoints, times, coefficients):
    #     # Evaluate the trajectory at high-resolution time points
    #     t_eval = np.linspace(times[0], times[-1], 1000)
    #     y_eval = np.polyval(coefficients, t_eval)

    #     # Evaluate the velocity and acceleration at the same time points
    #     velocity_eval = np.polyval(np.polyder(coefficients, 1), t_eval)
    #     acceleration_eval = np.polyval(np.polyder(coefficients, 2), t_eval)
    #     jerk_eval = np.polyval(np.polyder(coefficients, 3), t_eval)
    #     # Create subplots
    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

    #     # Plot position
    #     ax1.scatter(times, waypoints, color='red', label='Waypoints')
    #     ax1.plot(t_eval, y_eval, label='Position', color='blue')
    #     ax1.set_ylabel('Position')
    #     ax1.legend()

    #     # Plot velocity
    #     ax2.scatter(times, waypoints, color='red', label='Waypoints')
    #     ax2.plot(t_eval, velocity_eval, label='Velocity', color='green')
    #     ax2.set_ylabel('Velocity')
    #     ax2.legend()

    #     # Plot acceleration
    #     ax3.scatter(times, waypoints, color='red', label='Waypoints')
    #     ax3.plot(t_eval, acceleration_eval, label='Acceleration', color='orange')
    #     ax3.set_xlabel('Time')
    #     ax3.set_ylabel('Acceleration')
    #     ax3.legend()

    #     # Plot jerk
    #     ax4.plot(t_eval, jerk_eval, label='Jerk', color='pink')
    #     ax4.set_xlabel('Time')
    #     ax4.set_ylabel('Jerk')
    #     ax4.legend()
    #     # Customize plot
    #     plt.suptitle('Minimum Snap Trajectory with Velocity and Acceleration')
    #     plt.show()
    
    def min_snap_plan(self, waypoints):
        # Example waypoints
        # waypoints = [(0.5, 2), (1.5, 5), (2.5, 8), (3.5, 2), (4.5, 10)]
        # waypoints = self.waypoints
        waypoints_x = waypoints[:,0]
        waypoints_y = waypoints[:,1]
        waypoints_z = waypoints[:,2]
        # t = np.arange(waypoints.shape[0])
        # time_constraints = [t[0], t[-1]]
        # t_scaled = np.linspace(t[0], t[-1], int(duration*self.ctrl_freq))
        waypoints = [waypoints_x, waypoints_y, waypoints_z]
        times = np.arange(len(waypoints_x))
        print("times:", times)
        optimal_coefficients = []
        for waypoints_ in waypoints:
            self.num_coefficients = (len(waypoints_) )*3
            print("num_coeff:", self.num_coefficients)
            initial_guess = np.random.rand(self.num_coefficients)
            # Bounds for the optimization variables
            bounds = [(-10, 10) for _ in range(self.num_coefficients)]
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': self.constraint_position, 'args': (times, waypoints_)},
                {'type': 'eq', 'fun': self.constraint_velocity, 'args': [times]},
                {'type': 'eq', 'fun': self.constraint_acceleration, 'args': [times]},
                {'type': 'eq', 'fun': self.constraint_jerk, 'args': [times]},
                {'type': 'ineq', 'fun': self.constraint_acceleration_max,'args': (times, waypoints_, 3)} # max_acc?
            ]
            result = minimize(self.objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_coefficients.append(result.x)
        return optimal_coefficients


        # # Initial guess for polynomial coefficients (random values)
        
        # CTRL_duration = int(duration*self.CTRL_FREQ)
        # initial_guess = np.random.rand(15)

        # # Times at which constraints are specified
        # time_constraints = [0, CTRL_duration]

        # # Bounds for the optimization variables
        # bounds = [(-10, 10) for _ in range(len(initial_guess))]



        # # Optimization
        # result = minimize(self.objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        # # Extracting the optimal coefficients
        # optimal_coefficients = result.x
        # print("Optimal Coefficients:", optimal_coefficients)
