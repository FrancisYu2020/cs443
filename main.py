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
import utils
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=32, help="batch size used to train the model", type=int)
parser.add_argument("--alpha", default=0.5, type=float, help="weight of cls loss, default 0.5")
parser.add_argument("--exp_id", default=0, type=str, help="the id of the experiment")
parser.add_argument("--max_epsilon", default=1, type=float, help="the beginning epsilon (max epsilon)")
parser.add_argument("--min_epsilon", default=0.1, type=float, help="the final epsilon after decay during training (min epsilon)")
parser.add_argument("--epsilon_decay_frames", default=1000000, type=int, help="the final epsilon (min epsilon)")
parser.add_argument("--buffer_size", default=1000000, type=int, help="buffer size (number of frames)")
parser.add_argument("--training_frames", default=10000000, type=int, help="number of frames for training")
parser.add_argument("--atari_game", default='Pong', games = ['BeamRider','Breakout','Enduro','Pong','Qbert','Seaquest','SpaceInvaders'],\
                     help="name of the atari game environment")
args = parser.parse_args()

if args.atari_game == 'SpaceInvaders':
    args.frames_per_state = 3
else:
    args.frames_per_state = 4

args.epsilon_decay_steps = args.epsilon_decay_frames // args.frames_per_state
args.buffer_size /= args.frames_per_state
args.training_steps = args.training_frames // args.frames_per_state

args.exp_name = f"{args.atari_game}_epoch{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"
args.checkpoint_root = os.path.join('checkpoint', args.exp_name)
if not os.path.exists(args.checkpoint_root):
    os.mkdir(args.checkpoint_root)
    
args.exp_name += f'_{args.exp_id}'
args.checkpoint_dir = os.path.join(args.checkpoint_root, args.exp_id)
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="cs443",
    
    # track hyperparameters and run metadata
    config={
    "task": args.task,
    "window_size": args.window_size,
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
env = gym.make(args.atari_game)
args.action_space = env.action_space
phi_transforms = utils.dataset.phi()

replay_buffer = utils.dataset.ReplayBuffer([], args.buffer_size) # initialize the replay buffer
dataloader = DataLoader(replay_buffer, batch_size=args.batch_size, shuffle=True)

# initialize Q state-action value network and the target Q network
Q_net = utils.models.DQN(args.action_space.n).to(device)
target_net = Q_net.copy()
optimizer = optim.RMSprop(Q_net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

step = 0 # current step number
episode_id = 0
episode_rewards = []
total_reward = 0

# main loop
while step < args.training_steps:
    episode_id += 1
    episode_reward = 0
    # Initialise sequence s1 = {x1} and preprocessed sequenced φ_1 = φ(s_1)
    phi = phi_transforms(env.reset()) # φ_0 = φ(s_0)
    phi = torch.vstack([phi.copy()] * args.frames_per_state)

    target_net.eval()

    # collect the trajectory for T timestamps or until terminal state
    for t in range(args.T):
        # With probability epsilon select a random action a_t otherwise select at = max_{a} Q^*(φ(s_t), a; θ)
        action = target_net.epsilon_greedy(phi, utils.get_epsilon(args, step), args.action_space)
        
        phi_next, reward, done = [], 0, 0
        # Execute action at in emulator and observe reward r_t and image x_{t+1}
        for _ in range(args.frames_per_state):
            next_frame, curr_reward, curr_done, _ = env.step(action)
            phi_next.append(phi_transforms(next_frame))
            reward += curr_reward
            done |= curr_done
         
        # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
        phi_next = torch.vstack(phi_next)

        # Set y_j = r_j for terminal φ_{j+1}; rj + γ max_{a_0} Q(φ_{j+1}, a_0; θ) for non-terminal φ_{j+1}
        if not done:
            reward += args.gamma * Q_net(phi_next).max().detach() # need to detach from the computation graph
        
        episode_reward += reward
        
        # Store transition (φt, at, rt, φt+1) in D
        replay_buffer.add((phi, action, reward, phi_next))

        # Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
        for s, a, r, s_next in dataloader:
            Q_pred = Q_net(s)[a]
            loss = F.mse_loss(Q_pred, r)
            # Perform a gradient descent step on (y_j - Q(φ_j , a_j ; θ))^2 according to equation 3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        
        step += 1
    
    target_net = Q_net.copy()

env.close()

wandb.finish()
