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
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=32, help="batch size used to train the model", type=int)
parser.add_argument("--gamma", default=0.999, type=float, help="discount factor gamma")
parser.add_argument("--alpha", default=0.99, type=float, help="discount factor gamma")
parser.add_argument("--exp_id", default='debug', type=str, help="the id of the experiment")
parser.add_argument("--max_epsilon", default=1, type=float, help="the beginning epsilon (max epsilon)")
parser.add_argument("--min_epsilon", default=0.1, type=float, help="the final epsilon after decay during training (min epsilon)")
parser.add_argument("--epsilon_decay_frames", default=10000, type=int, help="the final epsilon (min epsilon)")
parser.add_argument("--buffer_size", default=10000, type=int, help="buffer size (number of frames)")
parser.add_argument("--training_frames", default=40000, type=int, help="number of frames for training")
parser.add_argument("--episode_length", default=200, type=int, help="number of frames for training")
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

args.epsilon_decay_steps = args.epsilon_decay_frames // args.frames_per_state
args.buffer_size //= args.frames_per_state
args.training_steps = args.training_frames // args.frames_per_state

# args.epsilon_decay_steps = args.epsilon_decay_frames
# args.training_steps = args.training_frames
print(f'Total training steps: {args.training_steps}')

args.exp_name = f"{args.atari_game}_lr{args.lr}_wd{args.weight_decay}"
args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
if not os.path.exists(args.checkpoint_root):
    os.mkdir(args.checkpoint_root)
    
args.exp_name += f'_{args.exp_id}'
args.checkpoint_dir = os.path.join(args.checkpoint_root, args.exp_id)
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

if args.wandb_log:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="cs443",
    
        # track hyperparameters and run metadata
        config={
            "game": args.atari_game,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
    
        # experiment name
        name=args.exp_name
    )

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

'''
In these experiments, we used the RMSProp algorithm with minibatches of size 32. The behavior
policy during training was epsilon-greedy with epsilon annealed linearly from 1 to 0.1 over the first million
frames, and fixed at 0.1 thereafter. We trained for a total of 10 million frames and used a replay
memory of one million most recent frames.
'''

'''
Algorithm 1 Deep Q-learning with Experience Replay
'''
# Create the atari game environment
env = gym.make(args.atari_game, )
args.action_space = env.action_space
phi_transforms = utils.dataset.phi()

replay_buffer = utils.dataset.ReplayBuffer([], args.buffer_size) # initialize the replay buffer
dataloader = None

# initialize Q state-action value network and the target Q network
Q_net = utils.models.DQN(args.action_space.n, in_channels=4).to(device)
target_net = copy(Q_net)
# optimizer = torch.optim.AdamW(Q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.SGD(Q_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer = optim.RMSprop(Q_net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

step = 0 # current step number
episode_id = 0
total_reward = 0
# print(env.reset().dtype)

# main loop
while step < args.training_steps:
    episode_id += 1
    episode_reward = 0
    # Initialise sequence s1 = {x1} and preprocessed sequenced φ_1 = φ(s_1)
    phi = phi_transforms(Image.fromarray(env.reset())).to(device) # φ_0 = φ(s_0)
    phi = torch.vstack([deepcopy(phi)] * args.frames_per_state)

    target_net.eval()
    
    episode_length = 0
    # collect the trajectory for T timestamps or until terminal state
    t = 0
    while t < args.episode_length:
        episode_length += 1
        # With probability epsilon select a random action a_t otherwise select at = max_{a} Q^*(φ(s_t), a; θ)
        action = target_net.epsilon_greedy(phi.unsqueeze(0), utils.get_epsilon(args, step), args.action_space).to(device)
        
        phi_next, reward, done = [], torch.tensor(0.).to(device), 0
#         next_frame, reward, done, _ = env.step(action)
#         phi_next = phi_transforms(Image.fromarray(next_frame)).to(device)
        # Execute action at in emulator and observe reward r_t and image x_{t+1}
        for i in range(args.frames_per_state):
            next_frame, curr_reward, curr_done, _ = env.step(action)
            phi_next.append(phi_transforms(Image.fromarray(next_frame)))
            reward += curr_reward
            done |= curr_done
        reward = utils.normalize_reward(torch.tensor(reward, device=args.device))
         
        # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
        phi_next = torch.vstack(phi_next).to(device)

        episode_reward += reward

        # Set y_j = r_j for terminal φ_{j+1}; rj + γ max_{a_0} Q(φ_{j+1}, a_0; θ) for non-terminal φ_{j+1}
        if not done:
#             reward += args.gamma * target_net(phi_next.unsqueeze(0)).max().clone().detach() # need to detach from the computation graph
            reward += args.gamma * Q_net(phi_next.unsqueeze(0)).max().clone().detach() # need to detach from the computation graph
#         reward = (1 - args.alpha) * Q_net(phi.unsqueeze(0)).squeeze(0)[action] + args.alpha * reward

        # Store transition (φt, at, rt, φt+1) in D
        replay_buffer.add((phi, action, reward.clone().detach(), phi_next))
        phi = phi_next.clone().detach()

        # Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
        if not dataloader:
            dataloader = DataLoader(replay_buffer, batch_size=args.batch_size, shuffle=True)
        
        for s, a, r, s_next in dataloader:
            Q_pred = torch.gather(Q_net(s), 1, a.to(device).unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(Q_pred, r)
            # Perform a gradient descent step on (y_j - Q(φ_j , a_j ; θ))^2 according to equation 3
            optimizer.zero_grad()
            loss.backward()
#             for name, module in Q_net.named_modules():
#                 if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                     wandb.log({name: module.weight.norm()}, step=step)
            optimizer.step()
            break

#         try:
#             for s, a, r, s_next in dataloader:
# #             print(s.size(), a, r, s_next.size())
#                 Q_pred = torch.gather(Q_net(s), 1, a.to(device).unsqueeze(1)).squeeze(1)
#                 loss = F.smooth_l1_loss(Q_pred, r)
# #                 loss = F.mse_loss(Q_pred, r)
#                 # Perform a gradient descent step on (y_j - Q(φ_j , a_j ; θ))^2 according to equation 3
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 break
#         except:
#             print(step)
        
        if done:
            break
        
        t += 1
        step += 1
        
    total_reward += episode_reward
    average_reward = total_reward / episode_id
    if args.wandb_log:
        wandb.log({'episode length': episode_length, 'episode reward': episode_reward, 'average reward': average_reward}, step=step)
    print(f'In episode {episode_id}, episode length: {episode_length}, episode reward: {episode_reward:.2f}, average reward: {average_reward:.2f}')
    
    if episode_id % 100 == 0:
        torch.save(Q_net.state_dict(), os.path.join(args.checkpoint_dir, f'episode_{episode_id}.ckpt'))
        print('model saved')
    target_net = deepcopy(Q_net)

env.close()

if args.wandb_log:
    wandb.finish()
