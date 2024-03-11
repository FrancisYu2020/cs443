# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# import gym

# # Specify the environment ID
# # env_id = "CartPole-v1"
# env_id = 'BreakoutDeterministic-v4'

# # Number of environments to run in parallel
# num_envs = 4

# # Create vectorized environments
# # This will automatically create `num_envs` copies of the environment
# # and allow you to step them in parallel
# vec_env = make_vec_env(env_id, n_envs=num_envs)

# # To stack frames if required, e.g., for Atari games
# # vec_env = VecFrameStack(vec_env, n_stack=4)

# # Example of interacting with the vectorized environment
# obs = vec_env.reset()
# for _ in range(25000):
#     actions = [vec_env.action_space.sample() for _ in range(num_envs)]  # Sample random actions
#     obs, rewards, dones, infos = vec_env.step(actions)
#     # Process the observations, rewards, dones, infos if needed
#     # If any environments are done, you might want to reset them in some cases

# # Close the environments
# vec_env.close()

# import torch
# import utils.models

# ckpt = torch.load('checkpoint/SeaquestNoFrameskip-v4_lr0.0001_wd1e-06/0/episode_1.ckpt')
# model = utils.models.DQN(18)
# model.load_state_dict(ckpt['state_dict'])
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
# optimizer.load_state_dict(ckpt['optimizer'])
# print(model)
# for param_group in optimizer.param_groups:
#     lr = param_group['lr']
#     print(lr)
import wandb

# Start a new run
run = wandb.init(project="cs443")

# Access and print the run ID
print("Run ID:", run.id)


# After completing your tasks
wandb.finish()
