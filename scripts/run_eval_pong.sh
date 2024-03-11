#!/bin/bash
#SBATCH --job-name="pong_eval"
#SBATCH --output="server_log/pong_eval_%j.%N.out"
#SBATCH --partition=gpux1
#SBATCH --time=1

atari_game=PongNoFrameskip-v4
lr=0.0001
batch_size=32
weight_decay=0.000001
total_frames=40000
episode_length=-1
wandb_log=1
optimizer=RMSprop
exp_id=full

time python eval.py --lr $lr --batch_size $batch_size --weight_decay $weight_decay --total_frames $total_frames --episode_length $episode_length --wandb_log $wandb_log --exp_id $exp_id --atari_game $atari_game --optimizer $optimizer

