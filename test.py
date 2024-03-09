import gym
import torch

# Create the environment
# env = gym.make('Pong')  # 'Deterministic' version for consistent results
env = gym.make('BreakoutDeterministic-v4')  # 'Deterministic' version for consistent results

# Initialize the environment
state = env.reset()

# Display the initial game state
# env.render()

# Loop for a few steps
for _ in range(1000000):
    # Take a random action
    action = torch.tensor(1)  # Replace this with your action selection mechanism
    
    # Perform the action and get the new state, reward, done (whether the game is over), and info
    next_state, reward, done, info = env.step(action)
#     print(action, type(action))
    
    # Display the game state
#     env.render()
    
    if done:
        break

# Close the environment
env.close()
