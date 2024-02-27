from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class RLSDataset(Dataset):
    def __init__(self, data, label, roi, transform=None, normalize_roi=False):
        """
        Args:
            data: (N, C, H, W)
            label: 0 or 1 indicating whether this is a positive region or not
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.images = data
        self.label = label
        self.roi = roi
        if normalize_roi:
            self.roi = self.roi * 1.0 / data.shape[1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label, roi = self.images[idx].T, self.label[idx], self.roi[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, roi