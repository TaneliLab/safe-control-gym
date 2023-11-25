"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) interStepLearn (optional)
        4) interEpisodeLearn (optional)

"""
import numpy as np

from collections import deque

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

except ImportError:
    # PyTest import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
    from planing.min_snap_generator import MinimumSnapTrajectory
    from parameters.get_params import ParamAssign
    from trajectory_planning.traj_generator import TrajGenerator
    from trajectory_modify.modifier import Modifier
    from trajectory_modify.time_strecher import Stretcher
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu
    from .planing.min_snap_generator import MinimumSnapTrajectory
    from .parameters.get_params import ParamAssign
    from .trajectory_planning.traj_generator import TrajGenerator
    from .trajectory_modify.modifier import Modifier
    from .trajectory_modify.time_strecher import Stretcher

#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Constraints analysis
        pos_constraint = {'x':[-3,3], 'y':[-3,3], 'z':[-0.1,2]}

        # Assign different parameters according to difficulty level
        self.reseed_per_episode = False
        self.assigner = ParamAssign(initial_info, self.reseed_per_episode)
        adjustable_params = self.assigner.get_params()

        # Parameter identifier
        self.temp_target = np.array([initial_obs[0], initial_obs[2], initial_obs[4], 0., 0., 0., 0., 0., 0., 0.])
                    # [ref_x, ref_y, ref_z, ref_vx, ref_vy, ref_vz, ref_ax, ref_ay, ref_az, ref_yaw]
        # Trajectory planning(1st task)
        self.takeoff_flag = False
        if 'obstacles' in initial_info['gates_and_obs_randomization']:
            robust_radius = max(abs(initial_info['gates_and_obs_randomization']['obstacles'].high),
                                abs(initial_info['gates_and_obs_randomization']['obstacles'].low))
        else:
            robust_radius = 0.


        obstacle_geo = [initial_info['obstacle_dimensions']['height'], initial_info['obstacle_dimensions']['radius'] + robust_radius]
        gate_geo = [initial_info['gate_dimensions']['tall']['edge'], 0.05, 0.05]
        gate_height = [initial_info['gate_dimensions']['tall']['height'], initial_info['gate_dimensions']['low']['height']]
        
        start_pos = [initial_obs[0], initial_obs[2], initial_obs[4]]
        if use_firmware:
            start_height = np.average(gate_height)
            start_pos[2] = start_height
        self.start_pos = start_pos
        goal_pos = [initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]]
        self.stop_pos = goal_pos
        # complete traj_plan(used later)
        traj_plan_params = {"ctrl_time": initial_info["episode_len_sec"], "ctrl_freq": self.CTRL_FREQ, "gate_sequence_fixed": True,
                    "start_pos": start_pos, "stop_pos": goal_pos, "max_recursion_num": adjustable_params['max_recursion_num'],
                    "uav_radius": 0.075, "obstacle_geo": obstacle_geo, "gate_geo": gate_geo, "accuracy": 0.01,
                    "gate_collide_angle": adjustable_params['gate_collide_angle'], "gate_height": gate_height,
                    "path_insert_point_dist_min": adjustable_params['path_insert_point_dist_min'],
                    "gate_waypoint_safe_dist": adjustable_params['gate_waypoint_safe_dist'],
                    "traj_max_vel": adjustable_params['traj_max_vel'], "traj_gamma": adjustable_params['traj_gamma']}


        # traj_plan_params = {"ctrl_time": initial_info["episode_len_sec"], "ctrl_freq": self.CTRL_FREQ, "gate_sequence_fixed": True,
        #     "start_pos": start_pos, "stop_pos": goal_pos,
        #     "uav_radius": 0.075, "obstacle_geo": obstacle_geo, "gate_geo": gate_geo, "accuracy": 0.01,
        #     "gate_height": gate_height}

        # Call a function in module `example_custom_utils`.
        ecu.exampleFunction()
        trajGen = TrajGenerator(traj_plan_params, self.NOMINAL_GATES, self.NOMINAL_OBSTACLES)

        for _ in range(adjustable_params['replan_attempt_num']):
            flag = trajGen.trajectory_replan()
            if flag:
                print("\033[4;33;40mMore attempts on trajectory planning may be needed.\033[0m")
            else:
                print("\033[0;37;42mReplanning Done!\033[0m")
                break
        
        generator = trajGen.traj_generator
        self.waypoints = np.array(trajGen.path)
        
        # Trajectory strecher
        near_edge_flag = min(abs(self.stop_pos[0]-pos_constraint['x'][0]),abs(self.stop_pos[0]-pos_constraint['x'][1]),\
                            abs(self.stop_pos[1]-pos_constraint['y'][0]),abs(self.stop_pos[1]-pos_constraint['y'][1]),\
                            abs(self.stop_pos[2]-pos_constraint['z'][0]),abs(self.stop_pos[2]-pos_constraint['z'][1])) <= 0.3
                            # 0.3 is the maximum overshoot allowed
        t_scaled = trajGen.timestamp
        traj_resample_params = {"dt": self.CTRL_TIMESTEP, "time_stretch_horizon": adjustable_params['time_stretch_horizon'], 
                                "pass_gate_vel": adjustable_params['pass_gate_vel'],"max_vel": adjustable_params['traj_max_vel'],
                                "time_ahead_bias": adjustable_params['time_ahead_bias'], "add_target_pos": near_edge_flag}
        self.strecher = Stretcher(t_scaled, trajGen.pos_trajectory, trajGen.gates, traj_resample_params)
        self.pos_trajectory, self.vel_trajectory, self.acc_trajectory, self.yaw_trajectory = self.strecher.resample(generator)
        t_scaled = self.strecher.timestamp

        ref_x = self.pos_trajectory[:,0].copy()
        ref_y = self.pos_trajectory[:,1].copy()
        ref_z = self.pos_trajectory[:,2].copy()

        # Trajectory modifier
        modifier_params = {'hold_ratio': adjustable_params['hold_ratio']}
        self.modifier = Modifier(np.array(self.NOMINAL_GATES)[:,0:2],modifier_params)

        # Save all planning parameters
        self.param_bucket = {'traj_plan_params': traj_plan_params, 'traj_resample_params': traj_resample_params}
 

        # # Example: hardcode waypoints through the gates.
        # if use_firmware:
        #     waypoints = [(self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"])]  # Height is hardcoded scenario knowledge.
        # else:
        #     waypoints = [(self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])]
        # for idx, g in enumerate(self.NOMINAL_GATES):
        #     height = initial_info["gate_dimensions"]["tall"]["height"] if g[6] == 0 else initial_info["gate_dimensions"]["low"]["height"]
        #     if g[5] > 0.75 or g[5] < 0:
        #         if idx == 2:  # Hardcoded scenario knowledge (direction in which to take gate 2).
        #             waypoints.append((g[0]+0.3, g[1]-0.3, height))
        #             waypoints.append((g[0]-0.3, g[1]-0.3, height))
        #         else:
        #             waypoints.append((g[0]-0.3, g[1], height))
        #             waypoints.append((g[0]+0.3, g[1], height))
        #     else:
        #         if idx == 3:  # Hardcoded scenario knowledge (correct how to take gate 3).
        #             waypoints.append((g[0]+0.1, g[1]-0.3, height))
        #             waypoints.append((g[0]+0.1, g[1]+0.3, height))
        #         else:
        #             waypoints.append((g[0], g[1]-0.3, height))
        #             waypoints.append((g[0], g[1]+0.3, height))
        #     # waypoints.append((g[0], g[1], height))
        # waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])
        # print("edit_waypointsL:", waypoints)
        # make time same as min_snap_plan

        # self.waypoints = np.array(waypoints)
        # t = np.arange(self.waypoints.shape[0])
        # print("edit_timelength: ", t)
        # duration = 15
        # t_scaled = np.linspace(t[0], t[-1], int(duration*self.CTRL_FREQ))
        
        # mini_snap_traj_generator = MinimumSnapTrajectory(traj_plan_params, self.NOMINAL_GATES, self.NOMINAL_OBSTACLES)
        # optimal_coefficients = mini_snap_traj_generator.min_snap_plan(self.waypoints)
        # self.ref_x  = np.polyval(optimal_coefficients[0], t_scaled)
        # self.ref_y  = np.polyval(optimal_coefficients[1], t_scaled)
        # self.ref_z  = np.polyval(optimal_coefficients[2], t_scaled)
        # self.ref_velo_x = np.polyval(np.polyder(optimal_coefficients[0], 1), t_scaled)
        # self.ref_velo_y = np.polyval(np.polyder(optimal_coefficients[1], 1), t_scaled)
        # self.ref_velo_z = np.polyval(np.polyder(optimal_coefficients[2], 1), t_scaled)

        # self.ref_acc_x = np.polyval(np.polyder(optimal_coefficients[0], 1), t_scaled)
        # self.ref_acc_y = np.polyval(np.polyder(optimal_coefficients[1], 1), t_scaled)
        # self.ref_acc_z = np.polyval(np.polyder(optimal_coefficients[2], 1), t_scaled)
        # # Polynomial fit.
        # self.waypoints = np.array(waypoints)
        # deg = 6
        # t = np.arange(self.waypoints.shape[0])
        # fx = np.poly1d(np.polyfit(t, self.waypoints[:,0], deg))
        # fy = np.poly1d(np.polyfit(t, self.waypoints[:,1], deg))
        # fz = np.poly1d(np.polyfit(t, self.waypoints[:,2], deg))
        # duration = 15
        # t_scaled = np.linspace(t[0], t[-1], int(duration*self.CTRL_FREQ))
        # self.ref_x = fx(t_scaled)
        # self.ref_y = fy(t_scaled)
        # self.ref_z = fz(t_scaled)

        # if self.VERBOSE:
        #     # Plot trajectory in each dimension and 3D.
        #     plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        #     # Draw the trajectory on PyBullet's GUI.
        #     draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


        if self.VERBOSE:
            # Plot trajectory in each dimension and 3D.
            plot_trajectory(t_scaled, self.waypoints, ref_x, ref_y, ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, ref_x, ref_y, ref_z)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time*self.CTRL_FREQ)

        # #########################
        # # REPLACE THIS (START) ##
        # #########################

        # # Handwritten solution for GitHub's getting_stated scenario.

        # if iteration == 0:
        #     height = 1
        #     duration = 2

        #     command_type = Command(2)  # Take-off.
        #     args = [height, duration]

        # elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
        #     step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
        #     target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
        #     target_vel = np.array([self.ref_velo_x[step], self.ref_velo_y[step], self.ref_velo_z[step]]) # np.zeros(3)
        #     target_acc = np.array([self.ref_acc_x[step], self.ref_acc_y[step], self.ref_acc_z[step]]) # np.zeros(3)
        #     target_yaw = 0.
        #     target_rpy_rates = np.zeros(3)

        #     command_type = Command(1)  # cmdFullState.
        #     args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        # elif iteration == 20*self.CTRL_FREQ:
        #     command_type = Command(6)  # Notify setpoint stop.
        #     args = []

        # elif iteration == 20*self.CTRL_FREQ+1:
        #     x = self.ref_x[-1]
        #     y = self.ref_y[-1]
        #     z = 1.5 
        #     yaw = 0.
        #     duration = 2.5

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif iteration == 23*self.CTRL_FREQ:
        #     x = self.initial_obs[0]
        #     y = self.initial_obs[2]
        #     z = 1.5
        #     yaw = 0.
        #     duration = 6

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif iteration == 30*self.CTRL_FREQ:
        #     height = 0.
        #     duration = 3

        #     command_type = Command(3)  # Land.
        #     args = [height, duration]

        # elif iteration == 33*self.CTRL_FREQ-1:
        #     command_type = Command(-1)  # Terminate command to be sent once the trajectory is completed.
        #     args = []

        # else:
        #     command_type = Command(0)  # None.
        #     args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        # #########################
        # # ustc (START) ##
        # #########################
        takeoff_time = 2.0
        pos_delta = 0.22 # determine whether or not uav has reached appointed position

        temp_pos = np.array([obs[0], obs[2], obs[4]])
        temp_att = np.array(obs[6:9])

        if 'current_target_gate_id' in info and 'current_target_gate_pos' in info:
            next_gate_id = info['current_target_gate_id']
            next_gate_pos = np.array(info['current_target_gate_pos'][0:2])
            self.modifier.refresh_gate_info(next_gate_id, next_gate_pos, self.temp_target[0:2])

        if iteration == 0:
            # before taking off
            self.takeoff_flag = False
            height = self.start_pos[2]
            command_type = Command(2)  # Take-off.
            args = [height, takeoff_time]
            self.temp_target = np.concatenate((self.start_pos, [0., 0., 0., 0., 0., 0., 0.]))
        elif self.takeoff_flag is False:
            # taking off
            if iteration <= takeoff_time*self.CTRL_FREQ or np.linalg.norm(temp_pos - self.start_pos) > pos_delta:
                self.real_start_iteration = iteration + 1
            else:
                self.takeoff_flag = True
            command_type = Command(0)  # None.
            args = []
        else:
            # after takeoff
            step = iteration - self.real_start_iteration
            if step <= len(self.pos_trajectory) -1:
                target_pos = self.pos_trajectory[step].copy()
                target_vel = self.vel_trajectory[step].copy()
                target_acc = self.acc_trajectory[step].copy()
                target_yaw = self.yaw_trajectory[step].copy()
                target_rpy_rates = np.zeros(3)

                # target_pos += np.append(self.modifier.get_des_pos_bias(target_pos[0:2]), 0.0)
                # target_acc += self.identifier.reference_signal_bias()
                
                command_type = Command(1)  # cmdFullState.
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
                self.temp_target = np.concatenate((target_pos, target_vel, target_acc, [target_yaw]))
            elif step == len(self.pos_trajectory):
                command_type = Command(6)  # notify setpoint stop.
                args = []
            elif step == len(self.pos_trajectory) + 1:
                command_type = Command(5)  # goTo.
                args = [self.stop_pos, 0., 2.5, False]
                self.temp_target = np.concatenate((self.start_pos, [0., 0., 0., 0., 0., 0., 0.]))
            elif (step - len(self.pos_trajectory)) % (2.5*self.CTRL_FREQ) == 1:
                command_type = Command(5)  # goTo.
                args = [self.stop_pos, 0., 2.5, False]
                self.temp_target = np.concatenate((self.start_pos, [0., 0., 0., 0., 0., 0., 0.]))
            else:
                command_type = Command(0)  # None.
                args = []

        if done:
            self.takeoff_flag = False

        #########################
        # ustc (END) ####
        #########################


        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the IROS 2022 Safe Robot Learning competition.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    @timing_step
    def interStepLearn(self,
                       action,
                       obs,
                       reward,
                       done,
                       info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1

        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
