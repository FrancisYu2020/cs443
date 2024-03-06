import numpy as np

def sliding_window_view(arr, window_size, stride):
    """
    Create a view of `arr` with a sliding window of size `window_size` moved by `stride`.
    
    Parameters:
    - arr: numpy array of 1 dimension.
    - window_size: size of the sliding window.
    - stride: step size between windows.
    
    Returns:
    - A 2D numpy array where each row is a window.
    """
    n = arr.shape[0]
    num_windows = (n - window_size) // stride + 1
    indices = np.arange(window_size)[None, :] + stride * np.arange(num_windows)[:, None]
    return arr[indices]

# Example usage
arr = np.arange(100)  # A sample numpy array
window_size = 4  # Length of the window
stride = 2  # Step size

# Apply the sliding window view function
result = sliding_window_view(arr, window_size, stride)

print(result)
