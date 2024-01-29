#!/usr/bin/env python
"""
Usage: 
    python cmdFullStateCFFirmware.py <path/to/controller.py> config.yaml

"""
import pickle as pkl
import os, sys

cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))
from pathlib import Path

import argparse, yaml
import time
from math import atan2, asin

import numpy as np

from pycrazyswarm import *
import rospy
from geometry_msgs.msg import TransformStamped

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class ViconWatcher:

    def __init__(self):
        # rospy.init_node("playback_node")
        config_path = Path(__file__).resolve().parents[2] / "launch/crazyflies.yaml"
        assert config_path.exists(), "Crazyfly config file missing!"
        with open(config_path, "r") as f:
            config = yaml.load(f)
        assert len(config["crazyflies"]) == 1, "Only one crazyfly allowed at a time!"
        cf_id = "cf" + str(config["crazyflies"][0]["id"])
        self.vicon_sub = rospy.Subscriber(f"/vicon/{cf_id}/{cf_id}", TransformStamped,
                                          self.vicon_callback)
        self.pos = None
        self.rpy = None

    def vicon_callback(self, data):
        self.child_frame_id = data.child_frame_id
        self.pos = np.array([
            data.transform.translation.x,
            data.transform.translation.y,
            data.transform.translation.z,
        ])
        rpy = euler_from_quaternion(
            data.transform.rotation.x,
            data.transform.rotation.y,
            data.transform.rotation.z,
            data.transform.rotation.w,
        )
        self.rpy = np.array(rpy)


class ObjectWatcher:

    def __init__(self, object: str = ""):
        self.vicon_sub = rospy.Subscriber("/vicon/" + object + "/" + object, TransformStamped,
                                          self.vicon_callback)
        self.pos = None
        self.rpy = None

    def vicon_callback(self, data):
        self.child_frame_id = data.child_frame_id
        self.pos = np.array([
            data.transform.translation.x,
            data.transform.translation.y,
            data.transform.translation.z,
        ])
        rpy = euler_from_quaternion(
            data.transform.rotation.x,
            data.transform.rotation.y,
            data.transform.rotation.z,
            data.transform.rotation.w,
        )
        self.rpy = np.array(rpy)


def load_controller(path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    try:
        return controller_module.Controller, controller_module.Command
    except ImportError as e:
        raise e


def eval_token(token):
    """Converts string token to int, float or str.

    """
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token


if __name__ == "__main__":
    SCRIPT_START = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", type=str, help="path to controller file")
    parser.add_argument("config", type=str, help="path to course configuration file")
    parser.add_argument("--overrides", type=str, help="path to environment configuration file")
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.controller))

    Controller, Command = load_controller(args.controller)

    swarm = Crazyswarm("../../launch/crazyflies.yaml")
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    vicon = ViconWatcher()

    hi1 = ObjectWatcher("cf_hi1")
    hi2 = ObjectWatcher("cf_hi2")
    lo1 = ObjectWatcher("cf_lo1")
    lo2 = ObjectWatcher("cf_lo2")
    obs1 = ObjectWatcher("cf_obs1")
    obs2 = ObjectWatcher("cf_obs2")
    obs3 = ObjectWatcher("cf_obs3")
    obs4 = ObjectWatcher("cf_obs4")

    timeout = 10
    while vicon.pos is None or vicon.rpy is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    print("Vicon found.")

    init_pos = vicon.pos
    init_rpy = vicon.rpy

    config_path = Path(args.config).resolve()
    assert config_path.is_file(), "Config file does not exist!"
    with open(config_path, "r") as f:
        config = yaml.load(f)
    nominal_gates_pos_and_type = config["gates_pos_and_type"]
    for gate in nominal_gates_pos_and_type:
        if gate[3] != 0 or gate[4] != 0:
            raise ValueError("Gates can't have roll or pitch!")
    nominal_obstacles_pos = config["obstacles_pos"]

    # Create a safe-control-gym environment from which to take the symbolic models
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    config.quadrotor_config['ctrl_freq'] = 500
    env = make('quadrotor', **config.quadrotor_config)
    env_obs, env_info = env.reset()
    print(env_obs, env_info)

    # Override environment state and evaluate constraints
    env.state = [
        vicon.pos[0], 0, vicon.pos[1], 0, vicon.pos[2], 0, vicon.rpy[0], vicon.rpy[1], vicon.rpy[2],
        0, 0, 0
    ]
    cnstr_eval = env.constraints.get_values(env, only_state=True)

    init_info = {
        #
        'symbolic_model': env_info[
            'symbolic_model'
        ],  # <safe_control_gym.math_and_models.symbolic_systems.SymbolicModel object at 0x7fac3a161430>,
        'nominal_physical_parameters': {
            'quadrotor_mass': 0.03454,
            'quadrotor_ixx_inertia': 1.4e-05,
            'quadrotor_iyy_inertia': 1.4e-05,
            'quadrotor_izz_inertia': 2.17e-05
        },
        #
        'x_reference': [-0.5, 0., 2.9, 0., 0.75, 0., 0., 0., 0., 0., 0., 0.],
        'u_reference': [0.084623, 0.084623, 0.084623, 0.084623],
        #
        'symbolic_constraints': env_info[
            'symbolic_constraints'
        ],  # [<function LinearConstraint.__init__.<locals>.<lambda> at 0x7fac49139160>, <function LinearConstraint.__init__.<locals>.<lambda> at 0x7fac3a14a5e0>],
        #
        'ctrl_timestep': 0.03333333333333333,
        'ctrl_freq': 30,
        'episode_len_sec': 33,
        'quadrotor_kf': 3.16e-10,
        'quadrotor_km': 7.94e-12,
        'gate_dimensions': {
            'tall': {
                'shape': 'square',
                'height': 1.0,
                'edge': 0.45
            },
            'low': {
                'shape': 'square',
                'height': 0.525,
                'edge': 0.45
            }
        },
        'obstacle_dimensions': {
            'shape': 'cylinder',
            'height': 1.05,
            'radius': 0.05
        },
        'nominal_gates_pos_and_type': nominal_gates_pos_and_type,
        'nominal_obstacles_pos': nominal_obstacles_pos,
        # 'nominal_gates_pos_and_type': [[0.5, -2.5, 0, 0, 0, -1.57, 0], [2, -1.5, 0, 0, 0, 0, 1], [0, 0.2, 0, 0, 0, 1.57, 1], [-0.5, 1.5, 0, 0, 0, 0, 0]],
        # 'nominal_obstacles_pos': [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]],
        #
        'initial_state_randomization': env_info[
            'initial_state_randomization'
        ],  # Munch({'init_x': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_y': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_z': Munch({'distrib': 'uniform', 'low': 0.0, 'high': 0.02}), 'init_phi': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_theta': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_psi': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1})}),
        'inertial_prop_randomization': env_info[
            'inertial_prop_randomization'
        ],  # Munch({'M': Munch({'distrib': 'uniform', 'low': -0.01, 'high': 0.01}), 'Ixx': Munch({'distrib': 'uniform', 'low': -1e-06, 'high': 1e-06}), 'Iyy': Munch({'distrib': 'uniform', 'low': -1e-06, 'high': 1e-06}), 'Izz': Munch({'distrib': 'uniform', 'low': -1e-06, 'high': 1e-06})}),
        'gates_and_obs_randomization': env_info[
            'gates_and_obs_randomization'
        ],  # Munch({'gates': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'obstacles': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1})}),
        'disturbances': env_info[
            'disturbances'
        ],  # Munch({'action': [Munch({'disturbance_func': 'white_noise', 'std': 0.001})], 'dynamics': [Munch({'disturbance_func': 'uniform', 'low': [-0.1, -0.1, -0.1], 'high': [0.1, 0.1, 0.1]})]}),
        # COULD/SHOULD THESE BE None or {} INSTEAD?
        #
        'urdf_dir':
            None,  # '/Users/jacopo/GitHub/beta-iros-competition/safe_control_gym/envs/gym_pybullet_drones/assets',
        'pyb_client': None,  # 0,
        'constraint_values':
            cnstr_eval  # [-2.03390077, -0.09345386, -0.14960551, -3.96609923, -5.90654614, -1.95039449]
    }

    CTRL_FREQ = init_info['ctrl_freq']

    # Create controller.
    vicon_obs = [
        init_pos[0], 0, init_pos[1], 0, init_pos[2], 0, init_rpy[0], init_rpy[1], init_rpy[2], 0, 0,
        0
    ]
    ctrl = Controller(vicon_obs, init_info, True)

    # Initial gate.
    current_target_gate_id = 0

    # ---- log data start
    # print("press button to takeoff ----- ")
    # swarm.input.waitUntilButtonPressed()

    # ---- commands for log
    log_cmd = []
    last_drone_pos = vicon.pos.copy()  # Helper for determining if the drone has crossed a goal
    completed = False
    print(f"Setup time: {time.time() - SCRIPT_START:.3}s")
    START_TIME = time.time()
    while not timeHelper.isShutdown():
        curr_time = time.time() - START_TIME

        done = False  # Leave always false in sim2real

        # Override environment state and evaluate constraints
        env.state = [
            vicon.pos[0], 0, vicon.pos[1], 0, vicon.pos[2], 0, vicon.rpy[0], vicon.rpy[1],
            vicon.rpy[2], 0, 0, 0
        ]
        state_error = (env.state - env.X_GOAL) * env.info_mse_metric_state_weight
        cnstr_eval = env.constraints.get_values(env, only_state=True)
        if env.constraints.is_violated(env, c_value=cnstr_eval):
            # IROS 2022 - Constrain violation flag for reward.
            env.cnstr_violation = True
            cnstr_num = 1
        else:
            # IROS 2022 - Constrain violation flag for reward.
            env.cnstr_violation = False
            cnstr_num = 0

        # This only looks at the x-y plane, could be improved
        gate_dist = np.sqrt(
            np.sum((vicon.pos[0:2] - nominal_gates_pos_and_type[current_target_gate_id][0:2])**2))
        #
        current_target_gate_in_range = True if gate_dist < 0.45 else False
        # current_target_gate_in_range = False # Sim2real difference, potentially affects solutions

        info = {  # TO DO
            'mse': np.sum(state_error**2),
            #
            'collision': (None, False),  # Leave always false in sim2real
            #
            'current_target_gate_id': current_target_gate_id,
            'current_target_gate_in_range': current_target_gate_in_range,
            'current_target_gate_pos': nominal_gates_pos_and_type[current_target_gate_id]
                                       [0:6],  # "Exact" regardless of distance
            'current_target_gate_type': nominal_gates_pos_and_type[current_target_gate_id][6],
            #
            'at_goal_position': False,  # Leave always false in sim2real
            'task_completed': False,  # Leave always false in sim2real
            #
            'constraint_values':
                cnstr_eval,  # array([-0.02496828, -0.08704742, -0.10894883, -0.04954095, -0.09521148, -0.03313234, -0.01123093, -0.07063881, -2.03338112, -0.09301162, -0.14799449, -3.96661888, -5.90698838, -1.95200551]),
            'constraint_violation': cnstr_num  # 0
        }

        # We transform the position of the drone into the reference frame of the current goal.
        # Goals have to be crossed in the direction of the y-Axis (pointing from -y to +y).
        # Therefore, we check if y has changed from negative to positive. If so, the drone has
        # crossed the plane spanned by the goal frame.
        # We then check the box conditions for x and z coordinates. First, we linearly interpolate
        # to get the x and z coordinates of the intersection with the goal plane. Then we check if
        # the intersection is within the goal box.
        # Note that we need to recalculate the last drone position each time as the transform
        # changes if the goal changes

        goal_pos = nominal_gates_pos_and_type[current_target_gate_id][0:3]
        goal_rot = nominal_gates_pos_and_type[current_target_gate_id][3:6]
        # Transform into current gate frame.
        cos_goal, sin_goal = np.cos(goal_rot[2]), np.sin(goal_rot[2])
        last_dpos = last_drone_pos - goal_pos
        last_drone_pos_gate = np.array([
            cos_goal * last_dpos[0] - sin_goal * last_dpos[1],
            sin_goal * last_dpos[0] + cos_goal * last_dpos[1], last_dpos[2]
        ])
        drone_pos = vicon.pos.copy()
        dpos = drone_pos - goal_pos
        drone_pos_gate = np.array([
            cos_goal * dpos[0] - sin_goal * dpos[1], sin_goal * dpos[0] + cos_goal * dpos[1],
            dpos[2]
        ])
        if last_drone_pos_gate[1] < 0 and drone_pos_gate[1] > 0:  # Drone has passed the goal plane
            alpha = -last_drone_pos_gate[1] / (drone_pos_gate[1] - last_drone_pos_gate[1])
            x_intersect = alpha * (drone_pos_gate[0]) + (1 - alpha) * last_drone_pos_gate[0]
            z_intersect = alpha * (drone_pos_gate[2]) + (1 - alpha) * last_drone_pos_gate[2]
            if abs(x_intersect) < 0.45 and abs(z_intersect) < 0.45:
                current_target_gate_id += 1
        last_drone_pos = drone_pos
        if current_target_gate_id == len(nominal_gates_pos_and_type):  # Reached the end
            current_target_gate_id = -1
            at_goal_time = time.time()

        if current_target_gate_id == -1:
            goal_pos = np.array([env.X_GOAL[0], env.X_GOAL[2], env.X_GOAL[4]])
            print(
                f"{time.time() - at_goal_time:.4}s and {np.linalg.norm(vicon.pos[0:3] - goal_pos)}m away"
            )
            if np.linalg.norm(vicon.pos[0:3] - goal_pos) >= 0.15:
                print(f"First hit goal position in {curr_time:.4}s")
                at_goal_time = time.time()
            elif time.time() - at_goal_time > 2:
                print(f"Task Completed in {curr_time:.4}s")
                completed = True
        #####################################################################
        #####################################################################

        #####################################################################
        #####################################################################
        reward = 0  # TO DO (or not needed for sim2real?)
        # # Reward for stepping through the (correct) next gate.
        # if stepped_through_gate:
        #     reward += 100
        # # Reward for reaching goal position (after navigating the gates in the correct order).
        # if at_goal_pos:
        #     reward += 100
        # # Penalize by collision.
        # if currently_collided:
        #     reward -= 1000
        # # Penalize by constraint violation.
        # if cnstr_violation:
        #     reward -= 100
        #####################################################################
        #####################################################################

        vicon_obs = [
            vicon.pos[0], 0, vicon.pos[1], 0, vicon.pos[2], 0, vicon.rpy[0], vicon.rpy[1],
            vicon.rpy[2], 0, 0, 0
        ]
        command_type, args = ctrl.cmdFirmware(curr_time, vicon_obs, reward, done, info)

        # print(vicon.pos)
        # print(current_target_gate_id, gate_dist)

        # ---- save the cmd for logging
        log_cmd.append([curr_time, rospy.get_time(), command_type, args])

        # Select interface.
        print("CMD", command_type, "ARGS", args)
        if command_type == Command.FULLSTATE:
            # print(args)
            # args = [args[0], [0,0,0], [0,0,0], 0, [0,0,0]]
            cf.cmdFullState(*args)
        elif command_type == Command.TAKEOFF:
            cf.takeoff(*args)
        elif command_type == Command.LAND:
            cf.land(*args)
        elif command_type == Command.STOP:
            cf.stop()
        elif command_type == Command.GOTO:
            cf.goTo(*args)
        elif command_type == Command.NOTIFYSETPOINTSTOP:
            cf.notifySetpointsStop()
        elif command_type == Command.NONE:
            pass
        elif command_type == Command.FINISHED:
            break
        else:
            raise ValueError("[ERROR] Invalid command_type.")

        timeHelper.sleepForRate(CTRL_FREQ)

        if completed:
            break

    cf.land(0, 3)
    timeHelper.sleep(3.5)

    # ---- save the command as pkl
    # print(log_cmd)
    with open('../decode_pkl/cmd_test_arg_video1.pkl', 'wb') as f:
        pkl.dump(log_cmd, f)

    # print("press button to land...")
    # swarm.input.waitUntilButtonPressed()