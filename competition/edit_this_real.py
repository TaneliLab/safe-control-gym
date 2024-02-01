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
import time as realtime

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
# from aggressiveTrajectoryPlanner.globalplanner import Globalplanner
# from aggressiveTrajectoryPlanner.SplineFactory import TrajectoryGenerator
# from aggressiveTrajectoryPlanner.localReplanner import LocalReplanner

# flexible in control points, cost design updated
from flexibleTrajectoryPlanner.globalplanner import Globalplanner
from flexibleTrajectoryPlanner.SplineFactory import TrajectoryGenerator
from flexibleTrajectoryPlanner.onlinelocalReplanner import OnlineLocalReplanner
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
        
        # hyperparmeters 
        self.LC_Module = False
        self.Planner_Type = "replan"   #"classical", "replan", "only_init"
        self.sampleRate = 5
        self.init_flight_time = 14  # 10 with AMAX=8 infeasible

        self.gate_id_now = -99
        self.takeoffFlag = False
        
        self.takeOffTime = 1
        self.takeOffHeight = 1
        self.onflyHeight = 1   # for adaptive control test 

        self.completeFlag = False
        self.high2lowlevelFlag = True  # allow notifysetpoint command
        self.low2highlevelFlag = True
        self.takeoff = False
        self.takeoff_cmd = False
        self.land = False
        self.land_cmd = False

        # Call a function in module `example_custom_utils`.
        ecu.exampleFunction()

        # # Example: hardcode waypoints through the gates.
        # height_tall = 1.14
        # height_low = 0.55
        # if use_firmware:
        #     waypoints = [
        #         (self.initial_obs[0], self.initial_obs[2], height_tall)
        #     ]  # Height is hardcoded scenario knowledge.
        # else:
        #     waypoints = [(self.initial_obs[0], self.initial_obs[2],
        #                   self.initial_obs[4])]

        # for idx, g in enumerate(self.NOMINAL_GATES):
        #     if g[6] == 0:  #tall
        #         waypoints.append(
        #             (g[0], g[1],
        #              initial_info["gate_dimensions"]["tall"]["height"]))
        #     else:
        #         waypoints.append(
        #             (g[0], g[1],
        #              initial_info["gate_dimensions"]["low"]["height"]))

        # waypoints.append([
        #     initial_info["x_reference"][0], initial_info["x_reference"][2],
        #     initial_info["x_reference"][4]
        # ])

        # waypoints2 = waypoints[1:-1]
        # waypoints2 = np.array(waypoints2)

        # # Polynomial fit.
        # self.waypoints = np.array(waypoints)

        # # TOM Version
        # if self.Planner_Type == "classic":
        #     trajPlanner = TrajectoryPlanner(waypoints[0], waypoints[-1],
        #                                     waypoints2, self.NOMINAL_OBSTACLES)

        #     trajPlanner.optimizer()

        #     trajectory = trajPlanner.spline
        #     omegaTrajectory = trajPlanner.omega_spline

        if self.Planner_Type == "replan":
            # trajGen = TrajectoryGenerator(waypoints[0], waypoints[-1],
            #                               waypoints2, self.NOMINAL_OBSTACLES, self.sampleRate, self.init_flight_time)

            # Better way of Generator, plug waypoints into Trajectory Generator
            trajGen = TrajectoryGenerator(initial_obs, initial_info,
                                          self.sampleRate,
                                          self.init_flight_time)
            self.traj_waypoints = trajGen.waypoints

            trajectory = trajGen.spline  #init spline

            trajPlanner = Globalplanner(trajectory, initial_obs, initial_info,
                                        self.sampleRate)
            trajPlanner.optimizer()
            trajectory = trajPlanner.spline

        elif self.Planner_Type == "only_init":
            
            trajGen = TrajectoryGenerator(initial_obs, initial_info,
                                          self.sampleRate,
                                          self.init_flight_time)
            self.traj_waypoints = trajGen.waypoints

            trajectory = trajGen.spline  #init spline

            trajPlanner = Globalplanner(trajectory, initial_obs, initial_info,
                                        self.sampleRate)


        self.trajectory = copy.copy(trajectory)
        self.flight_duration = trajPlanner.t  # flight duration
        print("flight time plan:", self.flight_duration)

        timesteps = np.linspace(0, self.flight_duration,
                                int(self.flight_duration * self.CTRL_FREQ))

        self.p = trajectory(timesteps)
        self.v = trajectory.derivative(1)(timesteps)
        self.a = trajectory.derivative(2)(timesteps)

        # self.omega = omegaTrajectory(timesteps)

        self.ref_x = self.p.T[0]
        self.ref_y = self.p.T[1]
        self.ref_z = self.p.T[2]

        # for acc_command compensation:
        self.acc_ff = [0, 0, 0]

        if self.VERBOSE:
            # Plot trajectory in each dimension and 3D.
            plot_trajectory(timesteps, self.traj_waypoints, self.ref_x,
                            self.ref_y, self.ref_z)

            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.traj_waypoints, self.ref_x,
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
        # realtime.sleep(1)
        if self.ctrl is not None:
            raise RuntimeError(
                "[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False."
            )

        iteration = int(time * self.CTRL_FREQ)
        self.current_time = time
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handwritten solution for GitHub's getting_stated scenario.

        endpoint_freq = self.flight_duration + 1  # can not set as planned duration, seems will not stop

        # endpoint_freq = 9
        if not self.takeoff:
            self.takeoff = obs[4] > 0.8
            height = 1
            duration = self.takeOffTime - 0.2
            command_type = Command(2)  # Take-off cmd
            args = [height, duration]
            if self.takeoff_cmd:
                command_type = Command(0)  # None.
                args = []
            else:
                self.takeoff_cmd = True

        elif self.high2lowlevelFlag:
            print("Notify setpoint stop. highlevel->lowlevel")
            command_type = Command(6)  # Notify setpoint stop.
            args = []
            self.high2lowlevelFlag = False

        elif iteration >= self.takeOffTime * self.CTRL_FREQ + 1 and iteration < endpoint_freq * self.CTRL_FREQ:
            step = min(iteration - self.takeOffTime * self.CTRL_FREQ,
                       len(self.ref_x) - 1)

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

        # elif self.low2highlevelFlag and self.completeFlag:
        #     print("Notify setpoint stop. lowlevel->highlevel")
        #     command_type = Command(6)  # Notify setpoint stop.
        #     args = []
        #     self.low2highlevelFlag = False

        # elif self.completeFlag and not self.land:
        #     self.land = obs[4]<0.05
        #     height = 0.
        #     duration = 2
        #     command_type = Command(3)  # Land.
        #     args = [height, duration]
        #     if self.land_cmd:
        #         command_type = Command(0)  # None.
        #         args = []
        #     else:
        #         self.land_cmd = True


        # elif self.completeFlag and self.land:
        #     command_type = Command(-1)
        #     # Terminate command to be sent once the trajectory is completed.
        #     args = []

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
        # self.action_buffer.append(
        #     action)  # [0.0899749 , 0.0852309 , 0.11418897, 0.11787074]
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
            # and current_gate_id not in self.passed_gate_id

            # We have 0.5~0.6 distance
            # print("self.gate_id_now:", self.gate_id_now, "current_gate_id:",
            #       current_gate_id)
            # print("info['current_target_gate_pos']:",
            #       info['current_target_gate_pos'])
            # # print("obs:", obs[0], obs[2], obs[4])
            # dist = np.linalg.norm(
            #     np.array([
            #         info['current_target_gate_pos'][0] - obs[0],
            #         info['current_target_gate_pos'][1] - obs[2],
            #         info['current_target_gate_pos'][2] - obs[4]
            #     ]))
            # print("distanceToGate:", dist)
            # print("      ")
            # when self.gate_id_now != current_gate_id the replan triggered once
            # then will make gate_id_now = current_gate_id so not triggered again

            # --------------------------Online replan block-------------
            # if self.Planner_Type == "replan" and self.gate_id_now != current_gate_id:
            #     trajLocalPlanner = OnlineLocalReplanner(
            #         self.trajectory, self.sampleRate, current_gate_id,
            #         true_gate_pose, obs, self.current_time-self.takeOffTime)

            #     self.trajectory = trajLocalPlanner.optimizer()

            #     self.gate_id_now = current_gate_id
            #     timesteps = np.linspace(
            #         0, self.flight_duration,
            #         int(self.flight_duration * self.CTRL_FREQ))

            #     self.p = self.trajectory(timesteps)
            #     self.v = self.trajectory.derivative(1)(timesteps)
            #     self.a = self.trajectory.derivative(2)(timesteps)

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
