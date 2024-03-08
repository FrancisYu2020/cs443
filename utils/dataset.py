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

def get_cnn_transforms(downsize=(110, 84), crop_size=(84, 84), train=True):
    # cnn transformation
    if train:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=downsize),
            transforms.RandomCrop(size=crop_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=downsize),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform

class ReplayBuffer(Dataset):
    def __init__(self, quadruples, transform):
        """
        Args:
            quadruples: list of (s, a, r, s')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.quadruples = quadruples
        self.transform = transform
        
    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        s, a, r, s_next = self.quadruples[idx]
        s = torch.cat([self.transform(frame) for frame in s], dim=0)
        s_next = torch.cat([self.transform(frame) for frame in s_next], dim=0)
        return s, a, r, s_next