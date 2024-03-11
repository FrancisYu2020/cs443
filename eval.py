import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
from tqdm import tqdm
import utils.dataset
import utils.metrics
import utils.models
import wandb
from copy import copy, deepcopy
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--batch_size", default=32, help="batch size used to train the model", type=int)
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor gamma")
parser.add_argument("--exp_id", default='debug', type=str, help="the id of the experiment")
parser.add_argument("--total_frames", default=40000, type=int, help="number of frames for training/eval")
parser.add_argument("--episode_length", default=200, type=int, help="number of frames for training/eval")
parser.add_argument("--optimizer", default='RMSprop', type=str, choices=['RMSprop', 'SGD', 'Adam', 'AdamW'], help="Choose the optimizer")
parser.add_argument("--atari_game", default='SeaquestNoFrameskip-v4', \
                    choices=['PongNoFrameskip-v4','BreakoutNoFrameskip-v4',\
                             'SpaceInvadersNoFrameskip-v4','MsPacmanNoFrameskip-v4',\
                             'QbertNoFrameskip-v4','SeaquestNoFrameskip-v4',\
                             'BeamRiderNoFrameskip-v4'], help="name of the atari game environment")
# parser.add_argument("--atari_game", default='SeaquestDeterministic-v4', \
#                     choices=['PongDeterministic-v4','BreakoutDeterministic-v4',\
#                              'SpaceInvadersDeterministic-v4','MsPacmanDeterministic-v4',\
#                              'QbertDeterministic-v4','SeaquestDeterministic-v4',\
#                              'BeamRiderDeterministic-v4'], help="name of the atari game environment")
parser.add_argument("--wandb_log", default=1, type=int, help="whether to use wandb to log this experiment")
args = parser.parse_args()

if args.episode_length < 0:
    args.episode_length = float('inf')

if args.atari_game == 'SpaceInvaders':
    args.frames_per_state = 3
else:
    args.frames_per_state = 4

args.total_steps = args.total_frames // args.frames_per_state

# args.epsilon_decay_steps = args.epsilon_decay_frames
# args.total_steps = args.total_frames
print(f'Eval steps for each checkpoint: {args.total_steps}')

args.exp_name = f"{args.atari_game}_gamma{args.gamma}_lr{args.lr}_wd{args.weight_decay}_{args.optimizer}"
args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
if not os.path.exists(args.checkpoint_root):
    raise FileNotFoundError(f'checkpoint root {args.checkpoint_root} not found!')
    
args.exp_name += f'_{args.exp_id}'
args.checkpoint_dir = os.path.join(args.checkpoint_root, args.exp_id)
if not os.path.exists(args.checkpoint_dir):
    raise FileNotFoundError(f'checkpoint directory {args.checkpoint_dir} not found!')

if args.wandb_log:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="cs443",
    
        # track hyperparameters and run metadata
        config={
            "game": args.atari_game,
            "gamma": args.gamma,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "total_frames": args.total_frames,
            "episode_length": args.episode_length,
            "experiment_id": args.exp_id
        },
    
        # experiment name
        name=args.exp_name + '_eval'
    )

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# Create the atari game environment
env = gym.make(args.atari_game)
args.action_space = env.action_space
phi_transforms = utils.dataset.phi()

# initialize Q state-action value network and the target Q network
Q_net = utils.models.DQN(args.action_space.n, in_channels=4).to(device)
'''
The statistics were computed by running an epsilon-greedy policy with epsilon = 0.05 for 10000 steps.
'''
args.epsilon = 0.05

checkpoints = [ckpt for ckpt in sorted(os.listdir(args.checkpoint_dir)) if ckpt.startswith('episode')]
print(checkpoints)
best_checkpoint = 'best_model.ckpt'
episode_checkpoints = {int(ckpt.split('.')[0].split('_')[1]):ckpt for ckpt in checkpoints}

for episode_id in sorted(episode_checkpoints.keys()):
    # reset step number for each checkpoint
    step = 0
    episode_rewards, episode_lengths = [], []
    
    Q_net.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, episode_checkpoints[episode_id]))['state_dict'])
    Q_net.eval()

    # main loop
    while step < args.total_steps:
        episode_reward, episode_length = 0, 0
    
        # Initialise sequence s1 = {x1} and preprocessed sequenced φ_1 = φ(s_1)
        phi = phi_transforms(Image.fromarray(env.reset())).to(device) # φ_0 = φ(s_0)
        phi = torch.vstack([deepcopy(phi)] * args.frames_per_state)
    
        # collect the trajectory for T timestamps or until terminal state
        t = 0
        while t < args.episode_length:
            episode_length += 1
            # With probability epsilon select a random action a_t otherwise select at = max_{a} Q^*(φ(s_t), a; θ)
            action = Q_net.epsilon_greedy(phi.unsqueeze(0), args.epsilon, args.action_space).to(device)
        
            phi_next, reward, done = [], torch.tensor(0.), 0
    #         next_frame, reward, done, _ = env.step(action)
    #         phi_next = phi_transforms(Image.fromarray(next_frame)).to(device)
            # Execute action at in emulator and observe reward r_t and image x_{t+1}
            for i in range(args.frames_per_state):
                next_frame, curr_reward, curr_done, _ = env.step(action)
                phi_next.append(phi_transforms(Image.fromarray(next_frame)))
                reward += curr_reward
                done |= curr_done
#             reward = utils.normalize_reward(torch.tensor(reward, device=args.device))
         
            # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
            phi_next = torch.vstack(phi_next).to(device)

            episode_reward += reward

            # Store transition (φt, at, rt, φt+1) in D
            phi = phi_next.clone().detach()
        
            if done:
                break
        
            t += 1
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f'In episode {episode_id}, episode length: {episode_length}, episode reward: {episode_reward:.2f}')
    
    if args.wandb_log:
        wandb.log({'average_episode_length': np.array(episode_length).mean(), 'average_reward': np.array(episode_rewards).mean()}, step=episode_id)
        
env.close()

if args.wandb_log:
    wandb.finish()
