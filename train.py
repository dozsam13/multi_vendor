from data_reader import DataReader
from torch.utils.data import DataLoader
import sys
import torch
from ventricle_segmentation_dataset import VentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import utils


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


def plot_data(data1, label1, data2, label2, filename):
    plt.clf()
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.legend()
    plt.savefig(filename)


def calc_dice(model, loader_dev):
    s = 0
    for sample in loader_dev:
        img = sample['image']
        target = sample['target'].cpu().detach().numpy()
        predicted = model(img)
        predicted = torch.round(predicted).cpu().detach().numpy()
        s += calc_dice_for_img(predicted, target)
    return s / len(loader_dev)


def calc_dice_for_img(predicted, target):
    smooth = 1.

    pred_flat = predicted.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = np.dot(pred_flat, target_flat)

    return (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)


def run_train():
    path = sys.argv[1]
    data_reader = DataReader(path)

    x_train = data_reader.x[:int(len(data_reader.x) * 0.66)]
    y_train = data_reader.y[:int(len(data_reader.x) * 0.66)]
    x_test = data_reader.x[int(len(data_reader.x) * 0.66):]
    y_test = data_reader.y[int(len(data_reader.x) * 0.66):]

    batch_size = 15
    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45]),
        transforms.ToTensor()
    ])
    device = torch.device('cuda')
    dataset_train = VentricleSegmentationDataset(x_train, y_train, device, augmenter)
    loader_train = DataLoader(dataset_train, batch_size)
    loader_train_accuracy = DataLoader(dataset_train, 1)
    dataset_dev = VentricleSegmentationDataset(x_test, y_test, device)
    loader_dev = DataLoader(dataset_dev, batch_size)
    loader_dev_accuracy = DataLoader(dataset_dev, 1)

    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1)),
        nn.BatchNorm2d(3),
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )

    epochs = 10
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    train_losses = []
    dev_losses = []
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
        train_losses.append(train_loss / len(loader_train))
        dev_losses.append(calculate_loss(loader_dev, model, criterion))
        utils.progress_bar(epoch + 1, epochs, 50, prefix='Training:')
    plot_data(train_losses, 'train_losses', dev_losses, 'dev_losses', 'losses.png')
    model.eval()
    print("Train dice: ", calc_dice(model, loader_train_accuracy))
    print("Test dice: ", calc_dice(model, loader_dev_accuracy))
    pred_mask = torch.round(model(dataset_train[0]['image'].unsqueeze(0))).cpu().detach().numpy().reshape(256, 256)
    expected_mask = dataset_train[0]['target'].cpu().detach().numpy().reshape(256, 256)
    plt.imsave('mask.png', pred_mask)
    plt.imsave('mask_expected.png', expected_mask)
    plt.imsave('image.png', dataset_train[0]['image'][0, :, :].cpu().detach().numpy().reshape(256, 256))


if __name__ == '__main__':
    run_train()
