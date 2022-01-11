from torch.utils.data import Dataset
import torch


class SingleDomainDataset(Dataset):
    def __init__(self, images, targets, device, augmenter=lambda x, y: (x, y)):
        self.images = images
        self.targets = targets
        self.device = device
        self.augmenter = augmenter

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, cnt = self.augmenter(self.images[index], self.targets[index])
        sample = {
            'image': torch.tensor(img.reshape(1, 256, 256), dtype=torch.float, device=self.device),
            'target': torch.tensor(cnt.reshape(1, 256, 256), dtype=torch.float, device=self.device),
        }
        return sample