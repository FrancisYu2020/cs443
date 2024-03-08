import torch.nn as nn
import torch
import torchvision.models as models
from utils.resnet3d import *
import os

# 10: is just a placeholder for class inheritence
resnet_2d_models = {10: models.resnet18, 18: models.resnet18, 34: models.resnet34}

def get_model(architecture_name, num_classes, window_size):
    '''
    create new model to train
    '''
    if 'resnet' in architecture_name:
        dimension, architecture = architecture_name.split('-')
        if dimension[0] == '2':
            return RLS2DModel(num_classes, int(architecture[-2:]), window_size=window_size)
        else:
            return RLS3DModel(num_classes, int(architecture[-2:]), window_size=window_size)
    else:
        raise NotImplementedError("ViT model part not implemented!")

def load_model(checkpoint_path, num_classes, window_size):
    '''
    load existing model to evaluate
    checkpoint_path: the experiment name of the model
    '''
    architecture_name = checkpoint_path.split('/')[1].split('_')[-1]
    model = get_model(architecture_name, num_classes, window_size)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model

'''
Architecture part of the DQN paper: 

The input to the neural network consists is an 84 × 84 × 4 image produced by φ. 

The first hidden layer convolves 16 8 × 8 filters with stride 4 with the input image and applies a rectifier nonlinearity. 

The second hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity.

The final hidden layer is fully-connected and consists of 256 rectifier units. 

The output layer is a fully-connected linear layer with a single output for each valid action.
'''

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)
    
class DQN(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.conv1 = ConvBlock(4, 16, 8, 4)
        self.conv2 = ConvBlock(16, 32, 4, 2)
        self.fc = nn.Sequential(
            nn.Linear(81, 256),
            nn.ReLU()
        )
        self.output = nn.Linear(256, action_space)
                    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return self.output(x)

    def epsilon_greedy(self, epsilon, action_space):
        '''
        With probability epsilon select a random action a_t, otherwise select at = max_{a} Q^*(phi(s_t), a; theta)
        '''
        assert 0 <= epsilon <= 1, "epsilon must be a real value in [0, 1]"
        if torch.rand(1) <= epsilon:
            return torch.randint(0, action_space)
        else:
            return torch.argmax(self.forward())