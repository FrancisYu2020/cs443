from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

'''
Working directly with raw Atari frames, which are 210 × 160 pixel images with a 128 color palette,
can be computationally demanding, so we apply a basic preprocessing step aimed at reducing the
input dimensionality. The raw frames are preprocessed by first converting their RGB representation
to gray-scale and down-sampling it to a 110×84 image. The final input representation is obtained by
cropping an 84 × 84 region of the image that roughly captures the playing area. The final cropping
stage is only required because we use the GPU implementation of 2D convolutions from [11], which
expects square inputs. For the experiments in this paper, the function φ from algorithm 1 applies this
preprocessing to the last 4 frames of a history and stacks them to produce the input to the Q-function.
'''

def phi(downsize=(110, 84), crop_size=(84, 84)):
    # cnn transforms phi
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=downsize),
        transforms.CenterCrop(size=crop_size),
        transforms.ToTensor(),
#         transforms.Normalize((0.4914), (0.2023)),
    ])
    return transform

class ReplayBuffer(Dataset):
    def __init__(self, quadruples, capacity):
        """
        Args:
            quadruples: list of (s, a, r, s')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.quadruples = quadruples
        self.capacity = capacity
        self.curr_idx = 0
    
    def add(self, quadruple):
        # we maintain a circle list for the quadruples to remove very old samples
        if len(self.quadruples) < self.capacity:
            self.quadruples.append(quadruple)
        else:
            self.quadruples[self.curr_idx % self.capacity] = quadruple
        self.curr_idx += 1
        
    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        return self.quadruples[idx]