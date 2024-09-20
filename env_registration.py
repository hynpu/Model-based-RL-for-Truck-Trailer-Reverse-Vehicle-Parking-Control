# env_registration.py
import gymnasium as gym
from gymnasium.envs.registration import register

from vehicleModel.ttwr_steering_velocity import TtwrFullControlEnv
from vehicleModel.ttwr_steering import TtwrSteerControlEnv
from vehicleModel.ttwr_steering_path_follow import TtwrSteerPathFollowEnv

# Register your environment
render_modes = ["human"]  # Specify the rendering modes supported by your environment
register(
    id='TtwrFullControlEnv-v0',
    entry_point='vehicleModel.ttwr_steering_velocity:TtwrFullControlEnv',
)

register(
    id='TtwrSteerControlEnv-v0',
    entry_point='vehicleModel.ttwr_steering:TtwrSteerControlEnv',
)

register(
    id='TtwrSteerPathFollowEnv-v0',
    entry_point='vehicleModel.ttwr_steering_path_follow:TtwrSteerPathFollowEnv',
)

register(
    id='TtwrSteerObstacleEnv-v0',
    entry_point='vehicleModel.ttwr_steering_obstacle:TtwrSteerObstacleEnv',
)

register(
    id='TtwrSteerBaseEnv-v0',
    entry_point='vehicleModel.ttwr_steering_base:TtwrSteerBaseEnv',
)

