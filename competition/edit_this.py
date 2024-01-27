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
import copy
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
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

# switch using which planner
# from trajectoryPlanner.trajectoryPlanner import TrajectoryPlanner
from aggressiveTrajectoryPlanner.trajectoryPlanner import TrajectoryPlanner
from systemIdentification.kRLS import KernelRecursiveLeastSquares, KernelRecursiveLeastSquaresMultiDim

# New SplineFactory and localReplanner
from aggressiveTrajectoryPlanner.globalplanner import Globalplanner
from aggressiveTrajectoryPlanner.SplineFactory import TrajectoryGenerator
from aggressiveTrajectoryPlanner.localReplanner import LocalReplanner
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
                 verbose: bool = False):
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
        self.LC_Module = True
        self.Planner_Type = "classical"
        self.Planner_Type = "replan"
        self.takeoffFlag = False
        self.sampleRate = 2
        self.completeFlag = False
        self.low2highlevelFlag = True
        self.init_flight_time = 12
        # Call a function in module `example_custom_utils`.
        ecu.exampleFunction()

        # Example: hardcode waypoints through the gates.
        if use_firmware:
            waypoints = [(self.initial_obs[0], self.initial_obs[2],
                          initial_info["gate_dimensions"]["tall"]["height"])
                         ]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [(self.initial_obs[0], self.initial_obs[2],
                          self.initial_obs[4])]

        for idx, g in enumerate(self.NOMINAL_GATES):
            height = initial_info["gate_dimensions"]["tall"]["height"] if g[
                6] == 0 else initial_info["gate_dimensions"]["low"]["height"]
            waypoints.append((g[0], g[1], height))

        waypoints.append([
            initial_info["x_reference"][0], initial_info["x_reference"][2],
            initial_info["x_reference"][4]
        ])

        waypoints2 = waypoints[1:-1] # only keep gate waypoints

        waypoints2 = np.array(waypoints2)

        # print(waypoints)
        # Polynomial fit.
        self.waypoints = np.array(waypoints)
        deg = 6
        
        # TOM Version
        if self.Planner_Type == "classic":
            trajPlanner = TrajectoryPlanner(waypoints[0], waypoints[-1],
                                            waypoints2, self.NOMINAL_OBSTACLES)

            trajPlanner.optimizer()

            trajectory = trajPlanner.spline
            omegaTrajectory = trajPlanner.omega_spline

        elif self.Planner_Type == "replan":
            trajGen = TrajectoryGenerator(waypoints[0], waypoints[-1],
                                          waypoints2, self.NOMINAL_OBSTACLES, self.sampleRate, self.init_flight_time)
            traj = trajGen.spline  #init spline

            trajPlanner = Globalplanner(traj, waypoints[0], waypoints[-1],
                                        waypoints2, self.NOMINAL_OBSTACLES)
            trajPlanner.optimizer()
            trajectory = trajPlanner.spline

        self.trajectory = copy.copy(trajectory)
        duration = trajPlanner.t

        self.flight_duration = 15  # for test
        self.takeOffTime = 1
        self.takeOffHeight = 1
        self.onflyHeight = 1
        # takeoff_timesteps = np.array([0 for i in range(self.takeOffTime*self.CTRL_FREQ -1)])
        # onfly_timesteps = np.linspace(0, duration-self.takeOffTime, int((duration-self.takeOffTime)*self.CTRL_FREQ))
        # # print("takeoff_timesteps:", takeoff_timesteps)
        # # print("onfly_timesteps:", onfly_timesteps)
        # timesteps = np.concatenate((takeoff_timesteps, onfly_timesteps))

        self.flight_duration = trajPlanner.t  # flight duration
        print("flight time plan:", self.flight_duration)
        timesteps = np.linspace(0, self.flight_duration,
                                int(self.flight_duration * self.CTRL_FREQ))
        t_scaled = timesteps

        self.p = trajectory(timesteps)

        self.v = trajectory.derivative(1)(timesteps)
        self.a = trajectory.derivative(2)(timesteps)

        # self.omega = omegaTrajectory(timesteps)

        self.ref_x = self.p.T[0]
        self.ref_y = self.p.T[1]
        self.ref_z = self.p.T[2]

        # # ---------------testing with simple trajectories -------------
        # self.ref_x = np.sin(timesteps) + self.initial_obs[0]
        # self.ref_y = np.cos(timesteps) - 1 + self.initial_obs[2]
        # # constant height
        # # self.ref_z = np.array([self.onflyHeight for ii in range(len(timesteps))])
        # self.ref_z = 0.2*np.sin(timesteps) + self.onflyHeight

        # For plotting and Learn to Compensate
        # self.onfly_time = []
        # self.onfly_ref_x = []
        # self.onfly_ref_y = []
        # self.onfly_ref_z = []
        # self.onfly_obs_x = []
        # self.onfly_obs_y = []
        # self.onfly_obs_z = []
        # self.onfly_acc_z = []

        # for acc_command compensation:
        self.acc_ff = [0, 0, 0]

        if self.VERBOSE:
            # Plot trajectory in each dimension and 3D.
            plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y,
                            self.ref_z)

            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x,
                            self.ref_y, self.ref_z)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):
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
            raise RuntimeError(
                "[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False."
            )

        iteration = int(time * self.CTRL_FREQ)
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handwritten solution for GitHub's getting_stated scenario.

        endpoint_freq = self.flight_duration + 1  # can not set as planned duration, seems will not stop

        # endpoint_freq = 9
        if iteration == 0:
            height = self.takeOffHeight
            # duration = 2
            duration = self.takeOffTime - 0.2

            command_type = Command(2)  # Take-off.
            self.takeoffFlag = True
            args = [height, duration]
        elif iteration == self.takeOffTime * self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

        elif iteration >= self.takeOffTime * self.CTRL_FREQ + 1 and iteration < endpoint_freq * self.CTRL_FREQ:
            step = min(iteration - self.takeOffTime * self.CTRL_FREQ,
                       len(self.ref_x) - 1)
            # step = min(iteration*self.CTRL_FREQ, len(self.ref_x) -1)
            # record onfly reference and obs for plot
            # self.onfly_time.append(time)
            # self.onfly_ref_x.append(self.ref_x[step])
            # self.onfly_ref_y.append(self.ref_y[step])
            # self.onfly_ref_z.append(self.ref_z[step])
            # self.onfly_obs_x.append(obs[0])
            # self.onfly_obs_y.append(obs[2])
            # self.onfly_obs_z.append(obs[4])

            # target_pos = self.p[step]
            # -----------Testing Simple plan -------------------
            # target_pos = np.array(
            #     [self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            # target_vel = np.zeros(3)
            # target_acc = np.array([0.0, 0.0, 0.0])
            # ----------------------------------------------------
            target_pos = self.p[step]
            target_vel = self.v[step]
            target_acc = self.a[step]

            # LC compensate
            if self.LC_Module:
                print("LC module is activated")
                # Fx and Fy noise always tricky 
                # TODO: solve this and uncomment
                # target_acc[0] = self.acc_ff[0]
                # target_acc[1] = self.acc_ff[1]
                target_acc[2] = self.acc_ff[2]

            # self.onfly_acc_z.append(self.acc_ff[2])

            # target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            # target_vel = np.array([self.ref_dx[step], self.ref_dy[step], self.ref_dz[step]])
            # target_acc = np.array([self.ref_ddx[step], self.ref_ddy[step], self.ref_ddz[step]])
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)
            # target_rpy_rates = self.omega[step]

            command_type = Command(1)  # cmdFullState.
            args = [
                target_pos, target_vel, target_acc, target_yaw,
                target_rpy_rates
            ]

            if step == len(self.ref_x) - 1:
                self.completeFlag = True

        # (Optional) Design for making it return after reach the goal
                
        # elif iteration >= endpoint_freq * self.CTRL_FREQ and iteration<endpoint_freq*self.CTRL_FREQ +5 and self.low2highlevelFlag:
        #     print("iteration: ", iteration)
        #     print("setpoint stop")
        #     self.low2highlevelFlag = False
        #     command_type = Command(6)  # Notify setpoint stop.
        #     args = []

        # elif iteration == int((endpoint_freq+1) * self.CTRL_FREQ):
        #     x = self.ref_x[-1]
        #     y = self.ref_y[-1]
        #     z = 1.5  # send to high
        #     yaw = 0.
        #     duration = 1.5

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif iteration == int((endpoint_freq + 3) * self.CTRL_FREQ):
        #     print("sendToInit")
        #     x = self.initial_obs[0]
        #     y = self.initial_obs[2]
        #     z = 1.5
        #     yaw = 0.
        #     duration = 6

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif iteration == int((endpoint_freq + 9) * self.CTRL_FREQ) + 1:
        #     print("land")
        #     height = 0.
        #     duration = 2

        #     command_type = Command(3)  # Land.
        #     args = [height, duration]

        elif iteration == int((endpoint_freq + 12) * self.CTRL_FREQ):
            command_type = Command(-1)
            # Terminate command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self, time, obs, reward=None, done=None, info=None):
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
            raise RuntimeError(
                "[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True."
            )

        iteration = int(time * self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([
                self.ref_x[iteration], self.ref_y[iteration],
                self.ref_z[iteration]
            ])
        else:
            target_p = np.array(
                [self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    @timing_step
    def interStepLearn(self, args, action, obs, reward, done, info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            args (List contains 4 array): Most recent command
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(
            action)  # [0.0899749 , 0.0852309 , 0.11418897, 0.11787074]
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)
        pos_command = list(args[0])
        self.acc_ff = list(args[2])  # current acc command
        self.ref_buffer.append(pos_command)
        #########################
        # REPLACE THIS (START) ##
        #########################
        if self.interstep_counter > 1:
            # TODO: extend to 3dim
            # rls_kernel = KernelRecursiveLeastSquares(num_taps=60, delta=0.01, lambda_=0.99, kernel='poly', poly_c=1, poly_d=3)
            # observation = self.obs_buffer[-1][4]
            # desired_output = self.ref_buffer[-1][2]
            # self.acc_ff[2] = rls_kernel.update(self.acc_ff[2], observation, desired_output)

            # 3 dim case
            # rls_kernel = KernelRecursiveLeastSquaresMultiDim(num_dims=3, num_taps=60, delta=0.01, lambda_=0.99, kernel='poly', poly_c=1, poly_d=3)
            rls_kernel = KernelRecursiveLeastSquaresMultiDim(num_dims=3,
                                                             num_taps=60,
                                                             delta=0.01,
                                                             lambda_=0.99,
                                                             kernel='poly',
                                                             poly_c=1,
                                                             poly_d=3)
            observation = [
                self.obs_buffer[-1][0], self.obs_buffer[-1][2],
                self.obs_buffer[-1][4]
            ]
            desired_output = [
                self.ref_buffer[-1][0], self.ref_buffer[-1][1],
                self.ref_buffer[-1][2]
            ]
            self.acc_ff = rls_kernel.update(self.acc_ff, observation,
                                            desired_output)
            # print("acc_ff:", self.acc_ff)

        if info['current_target_gate_in_range']:
            true_gate_pose = info['current_target_gate_pos']
            current_gate_id = info['current_target_gate_id']
            # TODO: Design Replan
            # Simpliest: Move gate control point to new center, and return the new spline
            # Return to self.p, self.v, self.a
            if self.Planner_Type == "replan":
                trajLocalPlanner = LocalReplanner(self.trajectory, self.sampleRate, current_gate_id, true_gate_pose)
                trajectory = trajLocalPlanner.hardGateSwitch()
                if trajectory:
                    timesteps = np.linspace(0, self.flight_duration,
                                int(self.flight_duration * self.CTRL_FREQ))
                    self.p = trajectory(timesteps)
                    self.v = trajectory.derivative(1)(timesteps)
                    self.a = trajectory.derivative(2)(timesteps)
        
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
        # add a new buffer for reference signal
        self.ref_buffer = deque([], maxlen=self.BUFFER_SIZE)

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
