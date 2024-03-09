# CS443 Project Repo

[Author]: Hang Yu (hangy6)

CS443 project code repo for CS443

## Design Choice
1. transforms resize use bilinear / other types of interpolation? (currently bilinear)
2. transforms centercrop / random crop (currently random crop for train, center crop for val)
3. for minibatch sample replay, we use a circle python list to maintain the replay histories

## Difference
1. In our implementation, we use OpenAI gym for Atari games which only supports one frame each time while the original paper stacks 4 frames together as one state.
2. In original paper, timestep 1 is used to describe the initial state and the first quadruple collected is (s_1, a_1, r_1, s_2), while for our implementation convenience, we use timestep 0 for the initial state and the first quadruple collected is (s_0, a_0, r_0, s_1).


## TODO
1. implement the epsilon decay algorithm [done]
2. modify the code to use skip frame (4 for all except 3 for Space Invader) as mentioned in the paper