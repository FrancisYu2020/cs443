# from pynput import keyboard
import keyboard
import pyautogui
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import time

class Annotator:
    def __init__(self, global_start_timestamp, screenshot_width=None, \
                 top_margin=803, left_margin=200, right_margin=10, bottom=867, \
                 interval_length=60, gap_threshold=2, screen_change_threshold=0.01, \
                 csv_file_path='red_pixels_coordinates.csv') -> None:
        # initialize the object with the preset attributes and parameters
        self.global_start_timestamp = global_start_timestamp
        self.screenshot_width = screenshot_width
        self.top_margin = top_margin
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.bottom = bottom
        self.interval_length = interval_length
        self.gap_threshold = gap_threshold
        self.screen_change_threshold = screen_change_threshold
        self.csv_file_path = csv_file_path

        # initialize the screenshot width by taking a screenshot and check if screenshot width is not specified
        if self.screenshot_width is None:
            self.screenshot_width = np.array(pyautogui.screenshot()).shape[1]

    def get_current_timestamp(self, start_timestamp, x_coordinate):
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
        interval_ratio = (x_coordinate - self.left_margin) / (self.screenshot_width - self.left_margin - self.right_margin)
        global_second = start_h * 3600 + start_min * 60 + start_s + self.interval_length * interval_ratio
        h = global_second // 3600
        global_second -= h * 3600
        min = global_second // 60
        s = global_second - min * 60
        return [h, min, s]

    def get_screenshot_start_timestamp(self, screenshot_idx):
        '''
        function to get the current screenshot's start timestamp

        global_start_timestamp: the start timestamp at the very beginning, type - tuple (start_h, start_min, start_s)
        screenshot_idx: the frame idx of currently played recording, 0-indexed
        intervals: the time interval used in playing the recording

        return: start_h, start_min, start_s
        '''
        start_h, start_min, start_s = self.global_start_timestamp
        global_second = start_h * 3600 + start_min * 60 + start_s + self.interval_length * screenshot_idx
        start_h = global_second // 3600
        global_second -= start_h * 3600
        start_min = global_second // 60
        start_s = global_second - start_min * 60
        return [start_h, start_min, start_s]

    def compress_intervals(self, arr):
        '''
        function used to compress the consecutive intervals

        arr: the input index arr of the positive regions, e.g. [2,2,2,3,4,5,10,11,12]
        gap_threshold: the maximum gap between two intervals, if smaller than this gap, the two intervals are merged

        return: list of intervals, e.g. [[2, 5], [10, 12]]
        '''
        # Ensure the array is unique and sorted
        arr = np.unique(arr)
        # Find the differences between consecutive elements
        diff = np.diff(arr)
        # Indices where the difference is greater than 1
        split_indices = np.where(diff > self.gap_threshold)[0] + 1
        # if no intervals, we directly return empty list
        if not split_indices:
            return []
        # Split the array into sub-arrays where the difference between numbers is more than 1
        sub_arrays = np.split(arr, split_indices)
        # Convert sub-arrays into intervals [min, max]
        intervals = [ [sub_arr[0], sub_arr[-1]] for sub_arr in sub_arrays ]
        return intervals

    def intervals_to_timestamps(self, intervals, screenshot_idx):
        '''
        function to convert the intervals got from the screen to the actual timestamps

        intervals: list of pixel intervals, type -> List[List]

        return: np.ndarray (len(intervals), 6)
        '''
        start_timestamp = self.get_screenshot_start_timestamp(screenshot_idx)
        for i in range(len(intervals)):
            start_x, end_x = intervals[i]
            curr_start = self.get_current_timestamp(start_timestamp, start_x)
            curr_end = self.get_current_timestamp(start_timestamp, end_x)
            intervals[i] = curr_start + curr_end
        return np.array(intervals) if intervals else None
        

    def screen_changed(self, prev_screen, new_screen):
        '''
        function to check if the new screen is different from the previous screen

        prev_screen: cv2 bgr array of the old screen
        new_screen: cv2 bgr array of the current screen
        threshold: the lower bound of the percentage of pixel change for the frame change

        return: boolean value of whether the screen has changed
        '''
        # Calculate the difference
        diff = cv2.absdiff(prev_screen, new_screen)
        # Convert difference to a percentage change
        percent_changed = np.sum(diff) / np.prod(new_screen.shape)
        return percent_changed > self.screen_change_threshold

    def process_screenshot(self, screenshot, screenshot_idx):
        '''
        main function to get the positive sample time intervals

        screenshot: the cv2 RGB image array input of the currently processed screenshot
        csv_file_path: path the the database where we save the label data
        top_margin: the length of the top margin to focus on the region of interest
        left_margin: the length of the left margin to focus on the region of interest
        right_margin: the length of the right margin to focus on the region of interest

        return: None
        ''' 
        # Find red pixels
        height, width, _ = screenshot.shape
        red_pixels = screenshot[self.top_margin : self.bottom, self.left_margin : -self.right_margin, ...]

        # red_pixels = np.where(red_pixels[:, :, 2] == 234)
        red_pixels = np.where((red_pixels[:, :, 2] > 200) & (red_pixels[:, :, 1] < 100) & (red_pixels[:, :, 0] < 100))
        coordinates = red_pixels[1] + self.left_margin
        intervals = self.compress_intervals(coordinates)

        # Save to CSV
        df = pd.DataFrame(self.intervals_to_timestamps(intervals, screenshot_idx), columns=['start_h', 'start_min', 'start_s', 'end_h', 'end_min', 'end_s'])
        df.to_csv(self.csv_file_path, mode='a', index=False, header=not os.path.exists(self.csv_file_path))
        # print(f"Screenshot taken and red pixels identified. Data appended to {self.csv_file_path}.")

        # visualization
        for i in range(len(coordinates)):
            point_position = (coordinates[i], 45 + self.top_margin)
            if coordinates[i] < self.bottom:
                cv2.circle(screenshot, point_position, 3, (255, 0, 0), -1)

        # Display the modified screenshot
        cv2.imwrite('test.png', screenshot)


