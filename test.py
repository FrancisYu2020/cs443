# import timeit
# import pyautogui
# from PIL import ImageGrab
# import cv2
# import numpy as np

# # Define functions to test
# def take_screenshot_pyautogui():
#     return pyautogui.screenshot()

# def take_screenshot_pillow():
#     return ImageGrab.grab()

# img = np.array(take_screenshot_pyautogui())
# h, w, _ = img.shape
# y = 81
# cv2.line(img, (0, y), (w, y), color=(0, 0, 255), thickness=1)
# y += 18
# cv2.line(img, (0, y), (w, y), color=(0, 0, 255), thickness=1)
# xl, xr = 689, w - 1186
# print(xl, xr)
# cv2.line(img, (xl, 0), (xl, h), color=(0, 0, 255), thickness=1)
# cv2.line(img, (xr, 0), (xr, h), color=(0, 0, 255), thickness=1)
# cv2.imwrite('test.png', img)
# import os
# os.system('start test.png')
# # # Number of iterations to test
# # number_of_executions = 100



# # # Time the pillow screenshot function
# # pillow_time = timeit.timeit(take_screenshot_pillow, number=number_of_executions)
# # print(f'Pillow ImageGrab screenshot time for {number_of_executions} executions: {pillow_time}')

# # # Time the pyautogui screenshot function
# # pyautogui_time = timeit.timeit(take_screenshot_pyautogui, number=number_of_executions)
# # print(f'PyAutoGUI screenshot time for {number_of_executions} executions: {pyautogui_time}')

# import timm
# import torch

# def print_model_parameter_count(model_name):
#     model = timm.create_model(model_name, pretrained=True)
#     num_parameters = sum(p.numel() for p in model.parameters())
#     print(f"{model_name}: {num_parameters:,} parameters")

# # Example for vit_tiny_patch16_224 and vit_base_patch16_224
# print_model_parameter_count('vit_tiny_patch16_224')
# print_model_parameter_count('vit_base_patch16_224')


# import torch
# # Choose the `slow_r50` model 
# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
# # print(model.blocks[0])
# # print(model.blocks[1])
# # print(model.blocks.2)
# # print(model.blocks.3)
# # print(model.blocks.4)
# # print(model.blocks[5])
# import torch.nn as nn

# x = nn.Linear(256, 1).cuda()
# y = torch.randn(2, 256).cuda()
# print(x(y), y.dtype)

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="rls",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()