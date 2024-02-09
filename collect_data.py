from pynput import keyboard
import pyautogui
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

#TODO: finish streaming the screenshots

# Specify the file path for saving the CSV
csv_file_path = 'red_pixels_coordinates.csv'

def get_current_timestamp(start_timestamp, left_margin, right_offset, screenshot_width, x_coordinate, interval_length=60):
     '''
     function used to interpolate the current timestamp using the start timestamp and pixel's x coordinate

     start_timestamp: the start timestamp at the current screenshot frame, type - tuple (start_h, start_min, start_s)
     left_margin: pixels need to be ignored on the left margin
     right_offset: pixels need to be ignored on the right margin
     screenshot_width: the total pixel width of the screenshot image
     x_coordinate: the x pixel coordinate of the point of interest 

     return: h, min, s
     '''
     start_h, start_min, start_s = start_timestamp
     interval_ratio = (x_coordinate - left_margin) / (screenshot_width - left_margin - right_offset)
     global_second = start_h * 3600 + start_min * 60 + start_s + interval_length * interval_ratio
     h = global_second // 3600
     global_second -= h * 3600
     min = global_second // 60
     s = global_second - min * 60
     return h, min, s

def get_screenshot_start_timestamp(global_start_timestamp, screenshot_idx, interval_length=60):
     '''
     function to get the current screenshot's start timestamp

     global_start_timestamp: the start timestamp at the very beginning, type - tuple (start_h, start_min, start_s)
     screenshot_idx: the frame idx of currently played recording, 0-indexed
     intervals: the time interval used in playing the recording

     return: start_h, start_min, start_s
     '''
     start_h, start_min, start_s = global_start_timestamp
     global_second = start_h * 3600 + start_min * 60 + start_s + interval_length * screenshot_idx
     start_h = global_second // 3600
     global_second -= start_h * 3600
     start_min = global_second // 60
     start_s = global_second - start_min * 60
     return start_h, start_min, start_s

def compress_intervals(arr):
    # Ensure the array is unique and sorted
    arr = np.unique(arr)
    # Find the differences between consecutive elements
    diff = np.diff(arr)
    # Indices where the difference is greater than 1
    split_indices = np.where(diff > 2)[0] + 1
    # Split the array into sub-arrays where the difference between numbers is more than 1
    sub_arrays = np.split(arr, split_indices)
    # Convert sub-arrays into intervals [min, max]
    intervals = [ [sub_arr[0], sub_arr[-1]] for sub_arr in sub_arrays ]
    return intervals

# Function to handle key press
def on_press(key):
    # Check if the key is the "enter" key
        if key == keyboard.Key.enter:
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            # print(screenshot.shape)
            
            # Find red pixels
            height, width, _ = screenshot.shape
            top_margin, left_margin = 803, 200
            red_pixels = screenshot[top_margin:867, left_margin:-10, ...]

            # red_pixels = np.where(red_pixels[:, :, 2] == 234)
            red_pixels = np.where((red_pixels[:, :, 2] > 200) & (red_pixels[:, :, 1] < 100) & (red_pixels[:, :, 0] < 100))
            coordinates = list(zip(red_pixels[1] + left_margin, [45 + top_margin] * len(red_pixels[0])))
            print(np.array(coordinates).shape)
            print(compress_intervals(np.array(coordinates)[:, 0]))
            # coordinates = list(zip(red_pixels[1] + left_margin, red_pixels[0] + top_margin))

            # Save to CSV
            df = pd.DataFrame(np.array(coordinates).reshape(-1, 1), columns=['X'])
            df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
            print(f"Screenshot taken and red pixels identified. Data appended to {csv_file_path}.")
            print('Hello')

            for i in range(len(coordinates)):
                point_position = coordinates[i]
                # point_position = [coordinates[i][0], int(0.4 * height)]
                if coordinates[i][1] < 867:
                    # print(coordinates[i])
                # if 1:
                    cv2.circle(screenshot, point_position, 3, (255, 0, 0), -1)
            # print(len(coordinates), screenshot[868, 2234])
            
            # Display the modified screenshot
            cv2.imwrite('test.png', screenshot)
            return False

        # elif key.char == 'x':
        elif key == keyboard.Key.esc:
            # stop listener
            return False

# Listen for key press
# with keyboard.Listener(on_press=on_press) as listener:
#     listener.join()
        
import numpy as np
from PIL import ImageGrab
import cv2

def screen_changed(prev_screen, new_screen, threshold=0.01):
    # Calculate the difference
    diff = cv2.absdiff(prev_screen, new_screen)
    # Convert difference to a percentage change
    percent_changed = np.sum(diff) / np.product(new_screen.shape)
    return percent_changed > threshold

def process_screenshot(screenshot, top_margin=803, left_margin=200, right_margin=10, bottom=867):
    '''
    main function to get the positive sample time intervals

    screenshot: the cv2 RGB image array input of the currently processed screenshot
    top_margin: the length of the top margin to focus on the region of interest
    left_margin: the length of the left margin to focus on the region of interest
    right_margin: the length of the right margin to focus on the region of interest

    return: None
    ''' 
    # Find red pixels
    height, width, _ = screenshot.shape
    top_margin, left_margin = 803, 200
    red_pixels = screenshot[top_margin:bottom, left_margin:-right_margin, ...]

    # red_pixels = np.where(red_pixels[:, :, 2] == 234)
    red_pixels = np.where((red_pixels[:, :, 2] > 200) & (red_pixels[:, :, 1] < 100) & (red_pixels[:, :, 0] < 100))
    coordinates = list(zip(red_pixels[1] + left_margin, [45 + top_margin] * len(red_pixels[0])))
    print(np.array(coordinates).shape)
    print(compress_intervals(np.array(coordinates)[:, 0]))
    # coordinates = list(zip(red_pixels[1] + left_margin, red_pixels[0] + top_margin))

    # Save to CSV
    df = pd.DataFrame(np.array(coordinates).reshape(-1, 1), columns=['X'])
    df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
    print(f"Screenshot taken and red pixels identified. Data appended to {csv_file_path}.")
    print('Hello')

    for i in range(len(coordinates)):
        point_position = coordinates[i]
        if coordinates[i][1] < bottom:
            cv2.circle(screenshot, point_position, 3, (255, 0, 0), -1)
            
    # Display the modified screenshot
    cv2.imwrite('test.png', screenshot)

# Capture the initial screen
prev_screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

test_iter = 100
while test_iter:
    test_iter -= 1
    # Capture a new screen after some time
    # new_screen = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_BGR2GRAY)
    new_screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    
    # Check if the screen has changed significantly
    if screen_changed(prev_screen, new_screen):
        process_screenshot(new_screen)

    # Update the previous screen
    prev_screen = new_screen

    # Add a delay or a method to exit the loop if necessary


# plt.plot(x, y)
# plt.savefig('histogram.png')