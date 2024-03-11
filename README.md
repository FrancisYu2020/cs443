# CS443 Project Repo

[Author]: Hang Yu (hangy6)

CS443 project code repo for CS443

## Design Choice
1. transforms resize use bilinear / other types of interpolation? (currently bilinear)
2. transforms centercrop
3. for minibatch sample replay, we use a circle python list to maintain the replay histories

## Difference
1. In our implementation, we use OpenAI gym for Atari games which only supports one frame each time while the original paper stacks 4 frames together as one state. (currently trying stacked version with openai gym, seems not make sense, compare with single frame later)
2. In original paper, timestep 1 is used to describe the initial state and the first quadruple collected is (s_1, a_1, r_1, s_2), while for our implementation convenience, we use timestep 0 for the initial state and the first quadruple collected is (s_0, a_0, r_0, s_1).
3. Due to the computation resource limit, instead of using a buffer for storing as large as 1M frames, we store 160k frames in the replay buffer instead, this may affect the final results.
4. $\gamma$ is not specified in the paper, we use $\gamma = 0.99$ for all the environment in this implementation.
5. We also try single skip frame as input in our implementation, in that case, 4M frames are used for training.

## Instructions
To run the training, either run:

<code>python train.py [specify all arguments]</code>

or

<code>sh scripts/run_train_[Atari game name].sh </code>

for default training settings.

Similarly, for evaluation, either run: 

<code>python eval.py [specify all arguments]</code>

or

<code>sh scripts/run_eval_[Atari game name].sh </code>

for corresponding evaluation.

## version history
Current version follows exactly the original DQN paper. Might want to change to DDQN, rainbow DQN etc later.

## TODO
1. implement the epsilon decay algorithm [done]
2. modify the code to use skip frame (4 for all except 3 for Space Invader) as mentioned in the paper [done]
3. batch run the job and hyperparameter tuning. [partially done]
4. write the evaluation code and evaluate checkpoints. [done]