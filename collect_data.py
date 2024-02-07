from pynput import keyboard
import pyautogui
import numpy as np
import pandas as pd
import cv2

# Specify the file path for saving the CSV
csv_file_path = 'red_pixels_coordinates.csv'

# Function to handle key press
def on_press(key):
    try:
        # Check if the key is the "enter" key
        if key == keyboard.Key.enter:
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            
            # Find red pixels
            red_pixels = np.where((screenshot[:, :, 2] > 150) & (screenshot[:, :, 0] < 100) & (screenshot[:, :, 1] < 100))
            coordinates = list(zip(red_pixels[1], red_pixels[0]))  # Swap to get (x, y) format
            
            # Save to CSV
            df = pd.DataFrame(coordinates, columns=['X', 'Y'])
            df.to_csv(csv_file_path, mode='a', index=False, header=not pd.read_csv(csv_file_path).shape[0])
            print(f"Screenshot taken and red pixels identified. Data appended to {csv_file_path}.")
            
    except AttributeError:
        # Check if the key is "x"
        if key.char == 'x':
            # Stop listener
            return False

# Listen for key press
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
