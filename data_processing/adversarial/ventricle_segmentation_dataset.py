from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np


class MultiSourceDataset(Dataset):
    default_augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    def __init__(self, images, targets, vendors, device, augmenter=lambda x, y: (x, y)):
        self.images = images
        self.targets = targets
        self.device = device
        self.augmenter = augmenter
        self.vendors = vendors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        (img, target) = self.augmenter(self.images[index], self.targets[index])
        sample = {
            'image': torch.tensor(img.reshape(1, 256, 256), dtype=torch.float, device=self.device),
            'target': torch.tensor(target.reshape(1, 256, 256), dtype=torch.float, device=self.device),
            'vendor': torch.tensor(self.vendors[index], dtype=torch.long, device=self.device)
        }
        return sample
