from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np


class VentricleSegmentationDataset(Dataset):
    default_augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    def __init__(self, images, targets, device, augmenter=lambda x: x):
        self.images = images
        self.targets = targets
        self.device = device
        self.augmenter = augmenter

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        stacked_data = np.append(self.targets[index], np.repeat(self.images[index], 2, axis=2), axis=2).astype('uint8')

        img = self.augmenter(stacked_data)*255.0
        sample = {
            'image': torch.unsqueeze(img[1, :, :], 0).to(self.device),
            'target': torch.unsqueeze(img[0, :, :], 0).to(self.device),
        }
        return sample
