# import gym
# import torch

# # Create the environment
# # env = gym.make('Pong')  # 'Deterministic' version for consistent results
# env = gym.make('BreakoutDeterministic-v4')  # 'Deterministic' version for consistent results

# # Initialize the environment
# state = env.reset()

# # Display the initial game state
# # env.render()

# # Loop for a few steps
# for _ in range(10000):
#     # Take a random action
#     action = torch.tensor(1)  # Replace this with your action selection mechanism
    
#     # Perform the action and get the new state, reward, done (whether the game is over), and info
#     next_state, reward, done, info = env.step(action)
# #     print(action, type(action))
    
#     # Display the game state
# #     env.render()
    
#     if done:
#         state = env.reset()

# # Close the environment
# env.close()

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gym

# Specify the environment ID
# env_id = "CartPole-v1"
env_id = 'BreakoutDeterministic-v4'

# Number of environments to run in parallel
num_envs = 4

# Create vectorized environments
# This will automatically create `num_envs` copies of the environment
# and allow you to step them in parallel
vec_env = make_vec_env(env_id, n_envs=num_envs)

# To stack frames if required, e.g., for Atari games
# vec_env = VecFrameStack(vec_env, n_stack=4)

# Example of interacting with the vectorized environment
obs = vec_env.reset()
for _ in range(25000):
    actions = [vec_env.action_space.sample() for _ in range(num_envs)]  # Sample random actions
    obs, rewards, dones, infos = vec_env.step(actions)
    # Process the observations, rewards, dones, infos if needed
    # If any environments are done, you might want to reset them in some cases

# Close the environments
vec_env.close()
