from data_reader import DataReader
from torch.utils.data import DataLoader
import sys
import torch
from ventricle_segmentation_dataset import VentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


def calculate_loss(loader, model, criterion):
    loss_sum = 0.0
    for sample in loader:
        image = sample['image']
        target = sample['target']
        predicted = model(image)
        loss = criterion(predicted, target)
        loss_sum += loss.cpu().detach().numpy()
        del loss

    return loss_sum / len(loader)


def run_train():
    path = sys.argv[1]
    data_reader = DataReader(path)

    x_train = data_reader.x[:int(len(data_reader.x)*0.66)]
    y_train = data_reader.y[:int(len(data_reader.x)*0.66)]
    x_test = data_reader.x[int(len(data_reader.x)*0.66):]
    y_train = data_reader.y[int(len(data_reader.x)*0.66):]

    batch_size = 2
    device = torch.device('cpu')
    dataset = VentricleSegmentationDataset(x_train, y_train, device)
    loader_train = DataLoader(dataset, batch_size)
    dataset = VentricleSegmentationDataset(x_test, y_train, device)
    loader_dev = DataLoader(dataset, batch_size)

    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1)),
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        )

    epochs = 1
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters())
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        for index, sample in enumerate(loader_train):
            img = sample['image']
            target = sample['target']
            predicted = model(img)
            loss = criterion(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().numpy()
            break
        train_losses.append(train_loss / len(loader_train))
    print(train_losses)


if __name__ == '__main__':
    run_train()

