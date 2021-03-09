from torch.utils.data import Dataset
import torch


class VentricleSegmentationDataset(Dataset):
    def __init__(self, images, targets, device):
        self.images = images
        self.targets = targets
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {
            'image': torch.tensor(self.images[index], dtype=torch.float, device=self.device),
            'target': torch.tensor(self.targets[index], dtype=torch.float, device=self.device)
        }
        return sample
