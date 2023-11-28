from typing import Any, Dict, Tuple

import numpy as np
from scipy. interpolate import CubicSpline
from .ek_controller_config import EkControllerConfig
from .planning import plan_time_optimal_trajectory_through_gates, State, Limits, Cylinder, to_pose


class EkControllerImplSimple:

    def __init__(self,
                 config: EkControllerConfig,
                 ):
        self._arc_parametrization_tolerance = 1e-4
        self._evenly_spaced_segments = 90
        self._take_off_height = 0.8  # 0.4
        self._gate_waypoint_offset = 0.2

        self._config = config

        self._dt = self._config.ctrl_timestep

        # self._rate_estimator = RateEstimator(self._dt)

        # # No path recompilation for us this round
        # self._risk_adviser = RiskAdviser(forced_conservative_mode = True)

        self._start_pos = (
            self._config.initial_obs[0],
            self._config.initial_obs[2],
            self._config.initial_obs[4]
        )
        self._start_yaw = self._config.initial_obs[8]

        self._goal_pos = (
            self._config.x_reference[0],
            self._config.x_reference[2],
            self._config.x_reference[4]
        )
        self._goal_yaw = self._start_yaw

        self._flight_plans_cache = {}

        # self.reset_episode() # include risk evaluate
        self._configure_mode(self._config.nominal_gates_pose_and_type)

    def _configure_mode(self, gate_poses):
        # nominal_gates_pose_and_type=initial_info["nominal_gates_pos_and_type"],

        # if risk_profile not in self._flight_plans_cache:
        #     print("Optimizing new flight path")
        #     waypoints_arg, waypoints_pos, landmarks = self._calculate_waypoints(
        #         gate_poses)
        #     ref_x, ref_y, ref_z = self._calculate_reference_trajectory(
        #         waypoints_pos, waypoints_arg)
        #     sequencer = self._build_flight_sequence(
        #         waypoints_pos=waypoints_pos, waypoints_arg=waypoints_arg, landmarks=landmarks)
        #     self._flight_plans_cache[risk_profile] = (
        #         waypoints_arg, waypoints_pos, landmarks, ref_x, ref_y, ref_z, sequencer)
        # else:
        #     print("Using cached risk profile")
        print("Optimizing new flight path")
        self._waypoints_arg, self._waypoints_pos, self.landmarks = self._calculate_waypoints(
            gate_poses)
        self._ref_x, self._ref_y, self._ref_z = self._calculate_reference_trajectory(
            self._waypoints_pos, self._waypoints_arg)
        
            
        
    def get_waypoints(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._waypoints_arg, self._waypoints_pos

    def get_reference_trajectory(self) -> Tuple:
        return self._ref_x, self._ref_y, self._ref_z
    
    def _calculate_reference_trajectory(self, waypoints_pos, waypoints_arg):
        fx = CubicSpline(waypoints_arg, waypoints_pos[:, 0])
        fy = CubicSpline(waypoints_arg, waypoints_pos[:, 1])
        fz = CubicSpline(waypoints_arg, waypoints_pos[:, 2])
        t_scaled = np.linspace(
            waypoints_arg[0],
            waypoints_arg[-1],
            100)
        ref_x = fx(t_scaled)
        ref_y = fy(t_scaled)
        ref_z = fz(t_scaled)
        return ref_x, ref_y, ref_z

    def _calculate_gate_pos(self, x, y, yaw, type):
        tall_gate_height = self._config.gate_dimensions["tall"]["height"]
        low_gate_height = self._config.gate_dimensions["low"]["height"]
        height = tall_gate_height if type == 0 else low_gate_height
        return (x, y, height, 0, 0, yaw)
    
    def _calculate_waypoints(self, gate_poses) -> Tuple[np.ndarray, np.ndarray]:
        # Determine waypoints
        air_start_pos = (
            self._start_pos[0], self._start_pos[1], self._take_off_height)

        assert self._config.gate_dimensions["tall"]["shape"] == "square"
        assert self._config.gate_dimensions["low"]["shape"] == "square"

        rotation_offset = 1.57
        gates = []
        for gate in gate_poses:
            gates.append(
                self._calculate_gate_pos(
                    x=gate[0],
                    y=gate[1],
                    yaw=gate[5] + rotation_offset,
                    type=gate[6]),
            )

        assert self._config.obstacle_dimensions["shape"] == "cylinder"
        obstacle_height = self._config.obstacle_dimensions["height"]
        obstacle_radius = self._config.obstacle_dimensions["radius"]

        obstacles = []
        for obstacle in self._config.nominal_obstacles_pos:
            obstacles.append(
                Cylinder(obstacle[0:3], radius=obstacle_radius, height=obstacle_height))

        print("Calculating best path through gates, may take a few seconds...")

        path = plan_time_optimal_trajectory_through_gates(
            initial_state=State(
                position=np.array(air_start_pos),
                velocity=np.zeros(3)),
            final_state=State(
                position=np.array(self._goal_pos),
                velocity=np.zeros(3)),
            gate_poses=list(map(to_pose, gates)),
            acceleration_limits=Limits(
                # lower=-1 * np.ones(3),
                # upper=1 * np.ones(3),
                lower=-0.2 * np.ones(3),
                upper=0.2 * np.ones(3),
            ),
            velocity_limits=Limits(
                # lower=np.array([0.05, -np.pi/6, -np.pi/6]),
                # upper=np.array([2.00, np.pi/6, np.pi/6]),
                lower=np.array([0.02, -np.pi/12, -np.pi/12]),
                upper=np.array([1.00, np.pi/12, np.pi/12]),
            ),
            num_cone_samples=3,
            obstacles=obstacles,
        )

        waypoint_pos = []
        waypoint_arg = []
        waypoint_marks = []
        for length, position, landmarks in path.evenly_spaced_points(
            self._evenly_spaced_segments, self._arc_parametrization_tolerance
        ):
            waypoint_pos.append(position)
            waypoint_arg.append(length)
            waypoint_marks.append(landmarks)

        return np.array(waypoint_arg), np.array(waypoint_pos), waypoint_marks