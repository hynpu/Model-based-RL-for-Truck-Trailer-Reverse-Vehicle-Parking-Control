from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from gymnasium import logger, spaces
from typing import Tuple

import vehicleModel.ttwr_assets.ttwr_config as ttwr_config
from vehicleModel.ttwr_steering_base import TtwrSteerBaseEnv
import planning.dubins.dubins_trailer as dubins
import vehicleModel.ttwr_assets.ttwr_helpers as ttwr_helpers


class TtwrSteerPathFollowEnv(TtwrSteerBaseEnv):
    def __init__(self, render_mode: Optional[str] = None, render_skip_frames: Optional[int] = 1):
        super().__init__(render_mode = render_mode, render_skip_frames = render_skip_frames)
        self.dubins_planner = dubins.dubinsPathPalnner(ttwr_config.path_res, ttwr_config.turning_radius)

        # Initialize the reference points for the trailer and host vehicle
        self.prev_trailer_ref_idx = 0
        self.prev_host_ref_idx = 0

        self.look_ahead = 5  # Default look-ahead distance
        # the lower the weight, the more important the error is, due to the exp function property
        self._err_states_weight = np.zeros((3, 3))
        self._err_states_weight[0, 0] = 0.4
        self._err_states_weight[1, 1] = 0.2
        self._err_states_weight[2, 2] = 0.1

        # Define observation space
        low_obs = np.array([-np.pi, -np.pi, -ttwr_config.max_deviation], dtype=np.float32)
        high_obs = np.array([np.pi, np.pi, ttwr_config.max_deviation], dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        # self.render_trajectory = True
        self.render_ref_point = True
    
    def reset(self, seed: Optional[int] = None, init_state: Optional[np.ndarray] = None, goal_state: Optional[np.ndarray] = None, init_v1=-ttwr_config.v1_max, direction = -1, path_ref_pnts: Optional[np.ndarray] = None):
        """
        Resets the environment and generates a new path.

        Returns:
        tuple: 
            - lat_error (float): The initial lateral error
            - info (dict): Additional information about the reset state
        """
        if init_state is None:
            init_state = np.array([40, 40, np.pi/4, 0], dtype=np.float32)
        if goal_state is None:
            goal_state = np.array([0, 0, 0, 0], dtype=np.float32)

        # Reset the parent environment
        state, info = super().reset(init_v1=init_v1, init_state=init_state, goal_state=goal_state, seed=seed)
        self.prev_trailer_ref_idx = 0
        self.prev_host_ref_idx = 0

        # Generate a new path using the Dubins planner if not provided
        if path_ref_pnts is None:
            path_ref_pnts = self.dubins_planner.plan_path(self.init_state, self.goal_state)
        else:
            # Check if the provided path needs extension
            init_deviation = np.linalg.norm(path_ref_pnts[0, 0:2] - self.state[0:2])
            if init_deviation > 4:
                self.dubins_planner.require_path_extension = False
                connection_ref_pnts = self.dubins_planner.plan_path(self.state[0:3], path_ref_pnts[0])
                path_ref_pnts = np.concatenate((connection_ref_pnts, path_ref_pnts), axis=0)

        # Set the trajectory in the parent class
        super().setTrajectories(path_ref_pnts)

        # Compute the initial lateral error
        lat_error = self._get_error_latErr()

        # Update info dictionary
        info.update({
            "init_state": self.state,
            "goal_state": self.goal_state,
            "path_length": len(self.path_ref_pnts)
        })

        return lat_error, info
    
    def _is_done(self, lat_error) -> Tuple[bool, dict]:
        """
        Check if the episode is done based on lateral error and jack-knife condition.

        Returns:
            Tuple[bool, dict]: 
                - done (bool): True if the episode is done, False otherwise.
                - done_info (dict): Dictionary containing information about why the episode ended.
        """
        done = False
        done_info = {
            "max_lat_error_exceeded": False,
            "jack_knifed": False,
            "goal_reached": False
        }

        # Check if lateral error is too large
        if abs(lat_error[0]) > np.pi / 2 or abs(lat_error[1]) > np.pi / 2 or abs(lat_error[2]) > ttwr_config.max_deviation:
            done = True
            done_info["max_lat_error_exceeded"] = True

        # Check for jack-knife condition
        if abs(self.state[3]) > ttwr_config.jackKnifeAngle:
            done = True
            done_info["jack_knifed"] = True

        # Check if goal is reached
        # compute the distance projection on goal_state[2] heading
        pos_diff = np.linalg.norm(self.state[:2] - self.goal_state[:2])
        long_diff = pos_diff * np.cos(self.goal_state[2])
        lat_diff = pos_diff * np.sin(self.goal_state[2])
        if abs(long_diff) < ttwr_config.long_dist_tolerance and abs(lat_diff) < ttwr_config.lat_dist_tolerance:
            done = True
            done_info["goal_reached"] = True

        return done, done_info
    
    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
        action: The steering angle of the host vehicle.

        Returns:
            - lat_error (float): The lateral error after taking the action
            - reward (float): The reward achieved by the action
            - done (bool): Whether the episode has ended
            - info (dict): Additional information about the step
        """
        # Execute the step in the parent environment
        _, _, done, _, info = super().step(action)        

        # Compute the lateral error based on the new state
        lat_error = self._get_error_latErr()

        # Calculate the reward based on the new state and episode status
        reward, done, done_info = self._compute_trajectory_reward(lat_error)

        return lat_error, reward, done, False, done_info

    def _compute_trajectory_reward(self, lat_error: np.ndarray[3]):
        """
        Computes the reward for the current state of the truck-trailer system.

        Args:
        lat_error (np.array): Current lat_error observation
        episode_success (bool): Whether the episode was successful
        episode_failed (bool): Whether the episode failed

        Returns:
        float: The total reward for the current state
        """
        done, done_info = self._is_done(lat_error)
        # If the episode is done, return the appropriate reward directly
        if done and not done_info["goal_reached"]:
            return -200, done, done_info
        if done and done_info["goal_reached"]:
            return 300, done, done_info

        # Calculate individual reward components
        phi_alignment_reward = -(np.sin(self.state[3]) ** 2)
        steering_reward = -(self.delta / ttwr_config.maxSteeringAngle)**2 
        deviation_reward = np.exp((( lat_error.T ).dot( -self._err_states_weight )).dot( lat_error ))

        # Define weights for each reward component
        weight_phi_alignment = 0.5
        weight_steering = 0.1
        weight_deviation = 0.4

        # Compute weighted sum of rewards
        total_reward = (
            weight_phi_alignment * phi_alignment_reward +
            weight_steering * steering_reward +
            weight_deviation * deviation_reward
        )

        return total_reward, done, done_info

    def _euclidean_distance(self, p1, p2):
        """
        Compute the Euclidean distance between two points.

        Args:
            p1 (np.ndarray): First point.
            p2 (np.ndarray): Second point.

        Returns:
            float: Euclidean distance between the two points.
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_closest_ref_pnt(self, vehicle_state, prev_idx):
        """
        Find the closest reference point to the given vehicle state.

        Args:
            vehicle_state (np.ndarray): Current state of the vehicle [x, y, heading].
            prev_idx (int): Index of the previously closest reference point.

        Returns:
            int: Index of the closest reference point.
        """
        cur_x, cur_y = vehicle_state[0], vehicle_state[1]
        min_dist = self._euclidean_distance(self.path_ref_pnts[prev_idx][0:2], [cur_x, cur_y])

        # Compute the distance from the current position to the previous reference point
        min_index = np.max([0, np.min([prev_idx, prev_idx - 10])])
        max_index = np.min([prev_idx + 50, len(self.path_ref_pnts) - 1])

        found_front_ref_pnt = False
        for i in range(prev_idx, max_index):
            cur_dist = self._euclidean_distance(self.path_ref_pnts[i][0:2], [cur_x, cur_y])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_index = i
                found_front_ref_pnt = True

        max_index = np.min([min_index + 50, len(self.path_ref_pnts) - 1])
        if not found_front_ref_pnt:
            for i in range(prev_idx, min_index, -1):
                cur_dist = self._euclidean_distance(self.path_ref_pnts[i][0:2], [cur_x, cur_y])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_index = i

        # Implement look-ahead distance
        if self.look_ahead > 0:
            min_index = min(min_index + self.look_ahead, len(self.path_ref_pnts) - 1)

        return min_index

    # get the closest reference point and compute the error, input is the current simulation step (index)
    def _get_error_latErr(self):
        """
        Get the lateral error for the trailer and host vehicle.

        Args:
            ttwr_states (np.ndarray): Current state of the trailer and host vehicle.

        Returns:
            np.ndarray: Lateral error for the trailer and host vehicle.
        """
        ttwr_states = self.full_state
        # Get the closest reference point for the trailer
        cur_trailer_ref_idx = self._get_closest_ref_pnt(ttwr_states[3:6], self.prev_trailer_ref_idx)
        self.ref_point = self.path_ref_pnts[cur_trailer_ref_idx]

        # Get the closest reference point for the host vehicle
        cur_host_ref_idx = self._get_closest_ref_pnt(ttwr_states[0:3], self.prev_host_ref_idx)

        # Calculate the heading errors for the trailer and host vehicle
        theta1_err = ttwr_helpers.wrapToPi((self.path_ref_pnts[cur_host_ref_idx][2] - ttwr_states[2]).item())
        theta2_err = ttwr_helpers.wrapToPi((self.path_ref_pnts[cur_trailer_ref_idx][2] - ttwr_states[5]).item())

        # Transform the distance vector from world coordinate to trailer coordinate
        rotation_mtx = np.array([[np.cos(ttwr_states[5]), np.sin(ttwr_states[5])],
                                  [-np.sin(ttwr_states[5]), np.cos(ttwr_states[5])]])
        dist_vec_tcs = self.path_ref_pnts[cur_trailer_ref_idx, 0:2] - ttwr_states[3:5]
        trailer_dist_err = rotation_mtx.dot(dist_vec_tcs)

        self.prev_trailer_ref_idx = cur_trailer_ref_idx
        self.prev_host_ref_idx = cur_host_ref_idx

        # Return the lateral error for the trailer and host vehicle
        # ref-theta1, ref-theta2, y_err
        return np.array([theta1_err, theta2_err, trailer_dist_err[1]], dtype=np.float32)