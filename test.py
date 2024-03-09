# import numpy as np

# def sliding_window_view(arr, window_size, stride):
#     """
#     Create a view of `arr` with a sliding window of size `window_size` moved by `stride`.
    
#     Parameters:
#     - arr: numpy array of 1 dimension.
#     - window_size: size of the sliding window.
#     - stride: step size between windows.
    
#     Returns:
#     - A 2D numpy array where each row is a window.
#     """
#     n = arr.shape[0]
#     num_windows = (n - window_size) // stride + 1
#     indices = np.arange(window_size)[None, :] + stride * np.arange(num_windows)[:, None]
#     return arr[indices]

# # Example usage
# arr = np.arange(100)  # A sample numpy array
# window_size = 4  # Length of the window
# stride = 2  # Step size

# # Apply the sliding window view function
# result = sliding_window_view(arr, window_size, stride)

# print(result)
import gym
import torch

# Create the environment
env = gym.make('BreakoutDeterministic-v4')  # 'Deterministic' version for consistent results

# Initialize the environment
state = env.reset()

# Display the initial game state
# env.render()

# Loop for a few steps
for _ in range(10000):
    # Take a random action
    action = torch.tensor(1)  # Replace this with your action selection mechanism
    
    # Perform the action and get the new state, reward, done (whether the game is over), and info
    next_state, reward, done, info = env.step(action)
    print(action, type(action))
    
    # Display the game state
#     env.render()
    
    if done:
        break

# Close the environment
env.close()
