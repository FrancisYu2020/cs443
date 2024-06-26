import torch.nn as nn
import torch
import torchvision.models as models
import os

# 10: is just a placeholder for class inheritence
resnet_2d_models = {10: models.resnet18, 18: models.resnet18, 34: models.resnet34}

def get_model(architecture_name, num_classes, window_size):
    '''
    create new model to train
    '''
    if 'resnet' in architecture_name:
        dimension, architecture = architecture_name.split('-')
        return 
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
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)
    
class DQN(nn.Module):
    def __init__(self, action_space, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 4 * in_channels, 8, 4),
            ConvBlock(4 * in_channels, 8 * in_channels, 4, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 648, 256), # (N, 8 * in_channels, 9, 9)
            nn.ReLU()
        )
        self.output = nn.Linear(256, action_space)
                    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return self.output(x)

    def epsilon_greedy(self, phi, epsilon, action_space):
        '''
        With probability epsilon select a random action a_t, otherwise select at = max_{a} Q^*(phi(s_t), a; theta)
        Args:
            phi: transformed state
            epsilon: greedy factor
            action_space: the action space of current environment
        '''
        assert 0 <= epsilon <= 1, "epsilon must be a real value in [0, 1]"
        if torch.rand(1) <= epsilon:
            return torch.tensor(action_space.sample())
        else:
            return torch.argmax(self.forward(phi))