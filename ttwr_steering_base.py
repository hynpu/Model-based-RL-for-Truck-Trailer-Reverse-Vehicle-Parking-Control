from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import vehicleModel.ttwr_assets.ttwr_config as ttwr_config
import vehicleModel.ttwr_assets.ttwr_helpers as ttwr_helpers
import pygame

class TtwrSteerBaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    TRUCK_COLOR = (0, 51, 102)
    TRAILER_COLOR = (255, 128, 0)
    WHEEL_COLOR = (64, 64, 64)
    HITCH_COLOR = (250, 127, 111)
    LINE_COLOR = (231, 218, 210)

    RECT_THICKNESS = 3
    WHEEL_THICKNESS = 4
    LINE_THICKNESS = 2

    def __init__(self, model_mode: Optional[str] = None, render_mode: Optional[str] = None, render_skip_frames: Optional[int] = 1):        
        """Load configurations from ttwr_config."""
        self.L1 = ttwr_config.L1
        self.L2 = ttwr_config.L2
        self.L3 = ttwr_config.L3
        self.dt = ttwr_config.dt
        self.phi_tolerance = ttwr_config.phi_tolerance
        self.theta_tolerance = ttwr_config.theta_tolerance
        self.lat_dist_tolerance = ttwr_config.lat_dist_tolerance
        self.long_dist_tolerance = ttwr_config.long_dist_tolerance

        """Initialize state variables."""
        self.direction = -1
        self.v1_abs = 0
        self.v1 = 0
        self.delta = 0
        self.steps = 0
        self.steps_beyond_terminated = None
        self.goal_state = ttwr_config.goal_state
        self.obstacles = None
        self.state = np.zeros(4)
        self.init_state = np.zeros(4)
        self.full_state = np.zeros(8)
        self.jack_knifed = False
        
        """Define observation and action spaces."""
        low_obs_state = np.array([ttwr_config.x_min, ttwr_config.y_min, -np.pi, -ttwr_config.jackKnifeAngle], dtype=np.float32)
        high_obs_state = np.array([ttwr_config.x_max, ttwr_config.y_max, np.pi, ttwr_config.jackKnifeAngle], dtype=np.float32)
        self.observation_space = spaces.Box(low_obs_state, high_obs_state, dtype=np.float32)

        act_min = np.array((-ttwr_config.maxSteeringAngle,), dtype=np.float32)
        act_max = np.array((ttwr_config.maxSteeringAngle,), dtype=np.float32)
        self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)

        """Setup rendering configurations."""
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_fps = 20
        self.render_skip_frames = render_skip_frames
        self.render_frame_count = 0
        self.render_obstacles = False
        self.render_trajectory = False
        self.render_ref_point = False
        self.path_ref_pnts = None
        self.ref_point = None
        self.x_min, self.x_max = -40/2, 60/2
        self.y_min, self.y_max = -50/2, 50/2

        self.model_mode = model_mode

    def reset(self, init_v1, init_state, goal_state, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.v1 = init_v1
        self.delta = 0
        self.obstacles = None
        self.path_ref_pnts = None
        self.ref_point = None
        self.state = np.array(init_state, dtype=np.float32)
        self.goal_state = np.array(goal_state, dtype=np.float32)
        self.init_state = self.state.copy()
        self.jack_knifed = False
        self.steps = 0
        self._compute_full_state()
        return self.state, {}

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the full state, initial state, and goal state."""
        return self.full_state, self.init_state, self.goal_state

    def _compute_full_state(self, trailer_state: Optional[np.ndarray] = None):
        """Compute the full state given the trailer state."""
        if trailer_state is None:
            trailer_state = self.state
        x2, y2, theta2, phi = trailer_state
        theta1 = theta2 - phi
        x1 = x2 + self.L2 * np.cos(theta1) + self.L3 * np.cos(theta2)
        y1 = y2 + self.L2 * np.sin(theta1) + self.L3 * np.sin(theta2)
        self.full_state = np.array([x1, y1, theta1, x2, y2, theta2, phi, self.v1, self.delta], dtype=np.float32)

    def set_full_state(self, full_state: np.ndarray):
        """Set the full state of the system."""
        self.full_state = full_state.copy()
        self.state = full_state[3:7].copy()
        self.goal_state = full_state[3:7].copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Perform one step in the environment."""
        x2, y2, theta2, phi = self.state

        # delta shall be within self.action_space
        delta = np.clip(action[0], self.action_space.low, self.action_space.high)
        self.delta = delta[0]

        # Update trailer state
        x2_dot = self.v1 * np.cos(phi) * (1 - self.L2 / self.L1 * np.tan(phi) * np.tan(self.delta)) * np.cos(theta2)
        y2_dot = self.v1 * np.cos(phi) * (1 - self.L2 / self.L1 * np.tan(phi) * np.tan(self.delta)) * np.sin(theta2)
        theta2_dot = -self.v1 * (np.sin(phi) / self.L3 + self.L2 / (self.L1 * self.L3) * np.cos(phi) * np.tan(self.delta))
        phi_dot = theta2_dot - self.v1 * np.tan(self.delta) / self.L1

        # Update state
        self.state = np.array([
            x2 + x2_dot * self.dt,
            y2 + y2_dot * self.dt,
            ttwr_helpers.wrapToPi(theta2 + theta2_dot * self.dt),
            phi + phi_dot * self.dt
        ])
        self.steps += 1
        self._compute_full_state()

        done = self._jack_knife()
        reward = 0

        if self.render_mode == "human":
            self.render()

        return self.state, reward, done, False, {}
    
    def _jack_knife(self):
        """Check if the vehicle is jack-knifed."""
        if np.abs(self.state[3]) > ttwr_config.jackKnifeAngle:
            self.jack_knifed = True
        return self.jack_knifed

    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def setObstables(self, obstacles: Optional[np.ndarray] = None):
        """Set obstacles for rendering."""
        self.render_obstacles = True
        if obstacles is not None:
            self.obstacles = obstacles

    def setTrajectories(self, path_ref_pnts: Optional[np.ndarray] = None):
        """Set trajectories for rendering."""
        self.render_trajectory = True
        if path_ref_pnts is not None:
            self.path_ref_pnts = path_ref_pnts

    def renderPathGeneration(self, full_state: np.ndarray, obstacles: Optional[np.ndarray] = None):
        """Render path generation."""
        self.full_state = full_state
        if obstacles is not None:
            self.obstacles = obstacles
        self.render()

    def render(self):
        """Render the current state of the environment."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return

        self.render_frame_count += 1
        if self.render_frame_count % self.render_skip_frames != 0:
            return

        screen_width, screen_height = self._setup_pygame()

        # save it for child classes
        self.screen_height = screen_height
        self.screen_width = screen_width

        self._draw_environment(screen_width, screen_height)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _setup_pygame(self):
        """Setup Pygame for rendering."""
        if self.render_mode == "human":
            screen_width, screen_height = 600 * 2, 400 * 2
        else:  # mode == "rgb_array"
            screen_width, screen_height = 600, 400

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        return screen_width, screen_height

    def transform_pygame_points(self, x, y):
        scale, map_x_min, map_y_min, offset_x, offset_y = self.pygame_visu_info
        screen_x = int(scale * (x - map_x_min) + offset_x)
        screen_y = int(scale * (y - map_y_min) + offset_y)
        return screen_x, screen_y

    def _draw_environment(self, screen_width, screen_height):
        """Draw the environment, including truck, trailer, and obstacles."""


        # Rendering code starts here
        x1, y1, theta1, x2, y2, theta2 = self.full_state[:6]

        map_x_min, map_x_max, map_y_min, map_y_max = self.x_min - 10, self.x_max + 10, self.y_min - 10, self.y_max + 10

        # Scale the coordinates to fit the screen
        scale = min(screen_width / (map_x_max - map_x_min),
                    screen_height / (map_y_max - map_y_min))
        offset_x = (screen_width - scale * (map_x_max - map_x_min)) / 2
        offset_y = (screen_height - scale * (map_y_max - map_y_min)) / 2

        self.pygame_visu_info = scale, map_x_min, map_y_min, offset_x, offset_y
        
        # host vehicle centroid point
        x1_cent = x1 + ttwr_config.L1/2 * np.cos(theta1)
        y1_cent = y1 + ttwr_config.L1/2 * np.sin(theta1)

        # host vehicle front reference point
        x1_front = x1 + ttwr_config.L1 * np.cos(theta1)
        y1_front = y1 + ttwr_config.L1 * np.sin(theta1)

        # hitch point
        hitch_x = x1 - ttwr_config.L2 * np.cos(theta1)
        hitch_y = y1 - ttwr_config.L2 * np.sin(theta1)

        # front wheels of host vehicle
        # compute left front wheel point using x1_front and y1_front
        x1_lf = x1_front - ttwr_config.host_width/2 * np.sin(theta1)
        y1_lf = y1_front + ttwr_config.host_width/2 * np.cos(theta1)
        # compute left front wheel after delta turn and wheel dimension
        x1_lf_frt = x1_lf + ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_lf_frt = y1_lf + ttwr_config.wheel_radius * np.sin(theta1+self.delta)
        x1_lf_rear = x1_lf - ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_lf_rear = y1_lf - ttwr_config.wheel_radius * np.sin(theta1+self.delta)

        # compute right front wheel point using x1_front and y1_front
        x1_rf = x1_front + ttwr_config.host_width/2 * np.sin(theta1)
        y1_rf = y1_front - ttwr_config.host_width/2 * np.cos(theta1)
        # compute right front wheel after delta turn and wheel dimension
        x1_rf_frt = x1_rf + ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_rf_frt = y1_rf + ttwr_config.wheel_radius * np.sin(theta1+self.delta)
        x1_rf_rear = x1_rf - ttwr_config.wheel_radius * np.cos(theta1+self.delta)
        y1_rf_rear = y1_rf - ttwr_config.wheel_radius * np.sin(theta1+self.delta)

        # rear wheels of host vehicle
        # compute left rear wheel point using x1_front and y1_front
        x1_lr = x1 - ttwr_config.host_width/2 * np.sin(theta1)
        y1_lr = y1 + ttwr_config.host_width/2 * np.cos(theta1)
        # compute left front wheel after delta turn and wheel dimension
        x1_lr_frt = x1_lr + ttwr_config.wheel_radius * np.cos(theta1)
        y1_lr_frt = y1_lr + ttwr_config.wheel_radius * np.sin(theta1)
        x1_lr_rear = x1_lr - ttwr_config.wheel_radius * np.cos(theta1)
        y1_lr_rear = y1_lr - ttwr_config.wheel_radius * np.sin(theta1)

        # compute left rear wheel point using x1_front and y1_front
        x1_rr = x1 + ttwr_config.host_width/2 * np.sin(theta1)
        y1_rr = y1 - ttwr_config.host_width/2 * np.cos(theta1)
        # compute left front wheel after delta turn and wheel dimension
        x1_rr_frt = x1_rr + ttwr_config.wheel_radius * np.cos(theta1)
        y1_rr_frt = y1_rr + ttwr_config.wheel_radius * np.sin(theta1)
        x1_rr_rear = x1_rr - ttwr_config.wheel_radius * np.cos(theta1)
        y1_rr_rear = y1_rr - ttwr_config.wheel_radius * np.sin(theta1)

        # wheels of trailer vehicle
        # compute left trailer wheel point using x2 and y2
        x2_lt = x2 - ttwr_config.trailer_width/2 * np.sin(theta2)
        y2_lt = y2 + ttwr_config.trailer_width/2 * np.cos(theta2)
        # compute left front wheel after delta turn and wheel dimension
        x2_lt_frt = x2_lt + ttwr_config.wheel_radius * np.cos(theta2)
        y2_lt_frt = y2_lt + ttwr_config.wheel_radius * np.sin(theta2)
        x2_lt_rear = x2_lt - ttwr_config.wheel_radius * np.cos(theta2)
        y2_lt_rear = y2_lt - ttwr_config.wheel_radius * np.sin(theta2)
        # compute right trailer wheel point using x2 and y2
        x2_rt = x2 + ttwr_config.trailer_width/2 * np.sin(theta2)
        y2_rt = y2 - ttwr_config.trailer_width/2 * np.cos(theta2)
        # compute right front wheel after delta turn and wheel dimension
        x2_rt_frt = x2_rt + ttwr_config.wheel_radius * np.cos(theta2)
        y2_rt_frt = y2_rt + ttwr_config.wheel_radius * np.sin(theta2)
        x2_rt_rear = x2_rt - ttwr_config.wheel_radius * np.cos(theta2)
        y2_rt_rear = y2_rt - ttwr_config.wheel_radius * np.sin(theta2)

        # compute rectangle corner points of host vehicle
        host_x_rect = np.array([x1_cent + ttwr_config.host_length/2 * np.cos(theta1) + ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent + ttwr_config.host_length/2 * np.cos(theta1) - ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent - ttwr_config.host_length/2 * np.cos(theta1) - ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent - ttwr_config.host_length/2 * np.cos(theta1) + ttwr_config.host_width/2 * np.sin(theta1), \
                                x1_cent + ttwr_config.host_length/2 * np.cos(theta1) + ttwr_config.host_width/2 * np.sin(theta1)])
        host_y_rect = np.array([y1_cent + ttwr_config.host_length/2 * np.sin(theta1) - ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent + ttwr_config.host_length/2 * np.sin(theta1) + ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent - ttwr_config.host_length/2 * np.sin(theta1) + ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent - ttwr_config.host_length/2 * np.sin(theta1) - ttwr_config.host_width/2 * np.cos(theta1), \
                                y1_cent + ttwr_config.host_length/2 * np.sin(theta1) - ttwr_config.host_width/2 * np.cos(theta1)])

        # compute rectangle corner points of host vehicle
        trailer_x_rect = np.array([x2 + ttwr_config.trailer_front_overhang * np.cos(theta2) + ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 + ttwr_config.trailer_front_overhang * np.cos(theta2) - ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 - ttwr_config.trailer_rear_overhang * np.cos(theta2) - ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 - ttwr_config.trailer_rear_overhang * np.cos(theta2) + ttwr_config.trailer_width/2 * np.sin(theta2), \
                                    x2 + ttwr_config.trailer_front_overhang * np.cos(theta2) + ttwr_config.trailer_width/2 * np.sin(theta2)])
        trailer_y_rect = np.array([y2 + ttwr_config.trailer_front_overhang * np.sin(theta2) - ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 + ttwr_config.trailer_front_overhang * np.sin(theta2) + ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 - ttwr_config.trailer_rear_overhang * np.sin(theta2) + ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 - ttwr_config.trailer_rear_overhang * np.sin(theta2) - ttwr_config.trailer_width/2 * np.cos(theta2), \
                                    y2 + ttwr_config.trailer_front_overhang * np.sin(theta2) - ttwr_config.trailer_width/2 * np.cos(theta2)])

        # Transform the coordinates
        host_x_rect, host_y_rect = zip(*[self.transform_pygame_points(x, y) for x, y in zip(host_x_rect, host_y_rect)])
        trailer_x_rect, trailer_y_rect = zip(*[self.transform_pygame_points(x, y) for x, y in zip(trailer_x_rect, trailer_y_rect)])
        hitch_x, hitch_y = self.transform_pygame_points(hitch_x, hitch_y)
        x1, y1 = self.transform_pygame_points(x1, y1)
        x2, y2 = self.transform_pygame_points(x2, y2)
        x1_lf_frt, y1_lf_frt = self.transform_pygame_points(x1_lf_frt, y1_lf_frt)
        x1_lf_rear, y1_lf_rear = self.transform_pygame_points(x1_lf_rear, y1_lf_rear)
        x1_rf_frt, y1_rf_frt = self.transform_pygame_points(x1_rf_frt, y1_rf_frt)
        x1_rf_rear, y1_rf_rear = self.transform_pygame_points(x1_rf_rear, y1_rf_rear)
        x1_lr_frt, y1_lr_frt = self.transform_pygame_points(x1_lr_frt, y1_lr_frt)
        x1_lr_rear, y1_lr_rear = self.transform_pygame_points(x1_lr_rear, y1_lr_rear)
        x1_rr_frt, y1_rr_frt = self.transform_pygame_points(x1_rr_frt, y1_rr_frt)
        x1_rr_rear, y1_rr_rear = self.transform_pygame_points(x1_rr_rear, y1_rr_rear)
        x2_lt_frt, y2_lt_frt = self.transform_pygame_points(x2_lt_frt, y2_lt_frt)
        x2_lt_rear, y2_lt_rear = self.transform_pygame_points(x2_lt_rear, y2_lt_rear)
        x2_rt_frt, y2_rt_frt = self.transform_pygame_points(x2_rt_frt, y2_rt_frt)
        x2_rt_rear, y2_rt_rear = self.transform_pygame_points(x2_rt_rear, y2_rt_rear)

        pygame.draw.polygon(self.surf, self.TRUCK_COLOR, list(zip(host_x_rect, host_y_rect)), self.RECT_THICKNESS)
        pygame.draw.polygon(self.surf, self.TRAILER_COLOR, list(zip(trailer_x_rect, trailer_y_rect)), self.RECT_THICKNESS)
        pygame.draw.circle(self.surf, self.HITCH_COLOR, (hitch_x, hitch_y), 2)
        pygame.draw.line(self.surf, self.LINE_COLOR, (hitch_x, hitch_y), (x1, y1), self.LINE_THICKNESS)
        pygame.draw.line(self.surf, self.LINE_COLOR, (hitch_x, hitch_y), (x2, y2), self.LINE_THICKNESS)
        pygame.draw.line(self.surf, self.WHEEL_COLOR, (x1_lf_frt, y1_lf_frt), (x1_lf_rear, y1_lf_rear), self.WHEEL_THICKNESS)
        pygame.draw.line(self.surf, self.WHEEL_COLOR, (x1_rf_frt, y1_rf_frt), (x1_rf_rear, y1_rf_rear), self.WHEEL_THICKNESS)
        pygame.draw.line(self.surf, self.WHEEL_COLOR, (x1_lr_frt, y1_lr_frt), (x1_lr_rear, y1_lr_rear), self.WHEEL_THICKNESS)
        pygame.draw.line(self.surf, self.WHEEL_COLOR, (x1_rr_frt, y1_rr_frt), (x1_rr_rear, y1_rr_rear), self.WHEEL_THICKNESS)
        pygame.draw.line(self.surf, self.WHEEL_COLOR, (x2_lt_frt, y2_lt_frt), (x2_lt_rear, y2_lt_rear), self.WHEEL_THICKNESS)
        pygame.draw.line(self.surf, self.WHEEL_COLOR, (x2_rt_frt, y2_rt_frt), (x2_rt_rear, y2_rt_rear), self.WHEEL_THICKNESS)

        # Draw the obstacles if they are provided
        if self.render_obstacles and self.obstacles is not None:
            for i in range(len(self.obstacles)):
                pygame.draw.circle(self.surf, (0, 0, 0), self.transform_pygame_points(self.obstacles[i][0], self.obstacles[i][1]), 5)

        if self.render_trajectory and self.path_ref_pnts is not None:
            for i in range(len(self.path_ref_pnts)):
                pygame.draw.circle(self.surf, (125, 125, 125), self.transform_pygame_points(self.path_ref_pnts[i][0], self.path_ref_pnts[i][1]), 1)

        if self.render_ref_point and self.ref_point is not None:
                pygame.draw.circle(self.surf, (0, 0, 0), self.transform_pygame_points(self.ref_point[0], self.ref_point[1]), 3)

        # Draw the goal state
        pygame.draw.circle(self.surf, (255, 0, 0), self.transform_pygame_points(self.goal_state[0], self.goal_state[1]), 5)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))