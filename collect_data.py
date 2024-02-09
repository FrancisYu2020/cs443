from pynput import keyboard
import pyautogui
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import time
import threading
from utils import *

#TODO: finish streaming the screenshots

# Define global variables here
running = True

def listen_for_exit():
    '''
    function for capturing the exit keyboard signal, default use 'esc' to exit
    '''
    global running
    print('Press "Esc" to exit...')
    while running:
        print(running)
        if keyboard.is_pressed('esc'):
            running = False
            break
        time.sleep(0.1)

# Specify the file path for saving the CSV
csv_file_path = 'red_pixels_coordinates.csv'
global_start_timestamp = [5, 36, 10]

annotator = Annotator(global_start_timestamp)

# Set up and start the listener thread for early exit
listener_thread = threading.Thread(target=listen_for_exit)
listener_thread.start()

# Capture the initial screen
prev_screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

epoch_count = 0

# Main loops
total_count = 5
while total_count:
    total_count -= 1

    # Capture a new screen after some time
    # new_screen = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_BGR2GRAY)
    new_screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    
    # Check if the screen has changed significantly
    if annotator.screen_changed(prev_screen, new_screen):
        annotator.process_screenshot(new_screen, csv_file_path)
        epoch_count += 1

    # Update the previous screen
    prev_screen = new_screen

    # prevent busy loop
    # time.sleep(0.5)

    # Add a delay or a method to exit the loop if necessary
print(f'total epoch counted: {epoch_count}')
running = False

listener_thread.join()
print('Program terminated')