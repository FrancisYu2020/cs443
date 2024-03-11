#!/bin/bash
#SBATCH --job-name="seaquest"
#SBATCH --output="server_log/seaquest_%j.%N.out"
#SBATCH --partition=gpux1
#SBATCH --time=12

atari_game=SeaquestDeterministic-v4
lr=0.0001
batch_size=32
weight_decay=0.000001
total_frames=4000000
epsilon_decay_frames=1000000
buffer_size=160000
episode_length=-1
wandb_log=1
max_epsilon=1
min_epsilon=0.1
optimizer=RMSprop
exp_id=single_frame

time python train.py --lr $lr --batch_size $batch_size --weight_decay $weight_decay --total_frames $total_frames --episode_length $episode_length --wandb_log $wandb_log --max_epsilon $max_epsilon --min_epsilon $min_epsilon --exp_id $exp_id --epsilon_decay_frames $epsilon_decay_frames --buffer_size $buffer_size --optimizer $optimizer --atari_game $atari_game

