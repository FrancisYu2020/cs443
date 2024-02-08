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

def get_current_timestamp(start_timestamp, left_offset, right_offset, screenshot_width, x_coordinate, interval_length=60):
     '''
     function used to interpolate the current timestamp using the start timestamp and pixel's x coordinate

     start_timestamp: the start timestamp at the current screenshot frame, type - tuple (start_h, start_min, start_s)
     left_offset: pixels need to be ignored on the left margin
     right_offset: pixels need to be ignored on the right margin
     screenshot_width: the total pixel width of the screenshot image
     x_coordinate: the x pixel coordinate of the point of interest 

     return: h, min, s
     '''
     start_h, start_min, start_s = start_timestamp
     interval_ratio = (x_coordinate - left_offset) / (screenshot_width - left_offset - right_offset)
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
            up_offset, left_offset = 803, 200
            red_pixels = screenshot[up_offset:867, left_offset:-10, ...]

            # red_pixels = np.where(red_pixels[:, :, 2] == 234)
            red_pixels = np.where((red_pixels[:, :, 2] > 200) & (red_pixels[:, :, 1] < 100) & (red_pixels[:, :, 0] < 100))
            coordinates = list(zip(red_pixels[1] + left_offset, [45 + up_offset] * len(red_pixels[0])))
            print(np.array(coordinates).shape)
            print(compress_intervals(np.array(coordinates)[:, 0]))
            # coordinates = list(zip(red_pixels[1] + left_offset, red_pixels[0] + up_offset))

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
    # try:
        # # Check if the key is the "enter" key
        # if key == keyboard.Key.enter:
        #     # Take a screenshot
        #     screenshot = pyautogui.screenshot()
        #     screenshot = np.array(screenshot)
        #     screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        #     print(screenshot.shape)
            
        #     # Find red pixels
        #     height, width, _ = screenshot.shape
        #     # red_pixels = np.where((screenshot[:, :, 2] > 150) & (screenshot[:, :, 0] < 100) & (screenshot[:, :, 1] < 100))
        #     red_pixels = screenshot[int(0.3 * height):height//2, ...]
            
        #     # max_column_pixel = red_pixels[:, :, 2].max(axis=0)
        #     # print(max_column_pixel.shape, max_column_pixel)
        #     # global x
        #     # global y
        #     # x, y = np.unique(max_column_pixel, return_counts=True)
        #     # cv2.imwrite('hello.png', red_pixels)
        #     # print(red_pixels.shape, red_pixels[:, :, 2].max(), red_pixels[:, :, 1].max(), red_pixels[:, :, 0].max())
        #     # red_pixels = (red_pixels[:, :, 2] > 250)
        #     # # red_pixels = (screenshot[:, :, 2] == 255) * (screenshot[:, :, 1] < 80) * (screenshot[:, :, 0] < 80)
        #     # # print(red_pixels.sum())
        #     # red_pixels = red_pixels.sum(axis=0)
        #     # red_pixels = np.where(red_pixels)
        #     # coordinates = red_pixels[0].reshape(-1, 1)  # Swap to get (x, y) format

        #     red_pixels = np.where(red_pixels[:, :, 2] == 249)
        #     coordinates = list(zip(red_pixels[1], red_pixels[0] + int(0.3 * height)))
        #     print(coordinates)

        #     # Save to CSV
        #     df = pd.DataFrame(coordinates.reshape(-1, 1), columns=['X'])
        #     df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
        #     print(f"Screenshot taken and red pixels identified. Data appended to {csv_file_path}.")
        #     print('Hello')

        #     # start_point = (0, int(0.3 * height))
        #     # end_point = (width, int(0.3 * height))
        #     # color = (255, 0, 0)  # Blue in BGR
        #     # thickness = 2
        #     # cv2.line(screenshot, start_point, end_point, color, thickness)
        #     # start_point = (0, height // 2)
        #     # end_point = (width, height // 2)
        #     # cv2.line(screenshot, start_point, end_point, color, thickness)
        #     # print(screenshot[816, 2200])
        #     # cv2.circle(screenshot, [2200,816], 30, (0, 0, 255), -1)
        #     for i in range(len(coordinates)//100):
        #         point_position = coordinates[i]
        #         # point_position = [coordinates[i][0], int(0.4 * height)]
        #         cv2.circle(screenshot, point_position, 30, (0, 0, 255), -1)
        #         print(coordinates[i])
        #     print(len(coordinates))
            
        #     # Display the modified screenshot
        #     cv2.imwrite('test.png', screenshot)
        #     return False
        #     # cv2.imshow("Screenshot with Horizontal Line", screenshot)
        #     # cv2.waitKey(0)  # Wait for a key press to close the displayed image
        #     # cv2.destroyAllWindows()

        # # elif key.char == 'x':
        # elif key == keyboard.Key.esc:
        #     # stop listener
        #     return False
            
    # except AttributeError:
    #     # Check if the key is "x"
    #     # if key.char == 'x':
    #     #     # Stop listener
    #     #     return False
    #     # raise NotImplementedError(f"unidentified keyboard input {key}")
    #     pass

# Listen for key press
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

# plt.plot(x, y)
# plt.savefig('histogram.png')