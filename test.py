import timeit
import pyautogui
from PIL import ImageGrab

# Define functions to test
def take_screenshot_pyautogui():
    pyautogui.screenshot()

def take_screenshot_pillow():
    ImageGrab.grab()

# Number of iterations to test
number_of_executions = 100



# Time the pillow screenshot function
pillow_time = timeit.timeit(take_screenshot_pillow, number=number_of_executions)
print(f'Pillow ImageGrab screenshot time for {number_of_executions} executions: {pillow_time}')

# Time the pyautogui screenshot function
pyautogui_time = timeit.timeit(take_screenshot_pyautogui, number=number_of_executions)
print(f'PyAutoGUI screenshot time for {number_of_executions} executions: {pyautogui_time}')