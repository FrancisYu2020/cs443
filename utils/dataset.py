from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class RLSDataset(Dataset):
    def __init__(self, data, label, transform=None):
        """
        Args:
            data_directory (string): Path to the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.images = data
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx].T, self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label