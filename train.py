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
import os
import pathlib
import img_warp


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


def split_data(ratio1, ratio2, data_x, data_y):
    n = len(data_x)
    test_x = data_x[int(n * ratio2):]
    test_y = data_y[int(n * ratio2):]
    data_dev_train_x = data_x[:int(n * ratio2)]
    data_dev_train_y = data_y[:int(n * ratio2)]
    n = len(data_dev_train_x)
    indices = list(range(len(data_dev_train_x)))
    np.random.shuffle(indices)
    train_indices = indices[:int(n * ratio1)]
    dev_indices = indices[int(n * ratio1):int(n * ratio2)]
    train_x = [data_dev_train_x[idx] for idx in train_indices]
    train_y = [data_dev_train_y[idx] for idx in train_indices]
    dev_x = [data_dev_train_x[idx] for idx in dev_indices]
    dev_y = [data_dev_train_y[idx] for idx in dev_indices]

    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)


def run_train():
    path = sys.argv[1]
    data_reader = DataReader(path)

    (x_train, y_train), (x_test, y_test), (_, _) = split_data(0.66, 0.99, data_reader.x, data_reader.y)

    batch_size = 15
    augmenter = transforms.Compose([
        img_warp.SineWarp(10),
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
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

    epochs = 20
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
    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_model.pth")
    torch.save(model.state_dict(), model_path)
    model.eval()
    print("Train dice: ", calc_dice(model, loader_train_accuracy))
    print("Test dice: ", calc_dice(model, loader_dev_accuracy))
    pred_mask = torch.round(model(dataset_dev[0]['image'].unsqueeze(0))).cpu().detach().numpy().reshape(256, 256)
    expected_mask = dataset_dev[0]['target'].cpu().detach().numpy().reshape(256, 256)
    plt.imsave('mask.png', pred_mask)
    plt.imsave('mask_expected.png', expected_mask)
    plt.imsave('image.png', dataset_dev[0]['image'][0, :, :].cpu().detach().numpy().reshape(256, 256))


def run_train_on_pretrained():
    path = sys.argv[1]
    data_reader = DataReader(path)

    (x_train, y_train), (x_test, y_test), (_, _) = split_data(0.66, 0.99, data_reader.x, data_reader.y)

    batch_size = 15
    augmenter = transforms.Compose([
        img_warp.SineWarp(10),
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
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

    state_d = torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_model.pth"))
    model.load_state_dict(state_d)

    epochs = 30
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
    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print("Train dice: ", calc_dice(model, loader_train_accuracy))
    print("Test dice: ", calc_dice(model, loader_dev_accuracy))
    pred_mask = torch.round(model(dataset_dev[0]['image'].unsqueeze(0))).cpu().detach().numpy().reshape(256, 256)
    expected_mask = dataset_dev[0]['target'].cpu().detach().numpy().reshape(256, 256)
    plt.imsave('mask.png', pred_mask)
    plt.imsave('mask_expected.png', expected_mask)
    plt.imsave('image.png', dataset_dev[0]['image'][0, :, :].cpu().detach().numpy().reshape(256, 256))


def eval():
    path = sys.argv[1]
    data_reader = DataReader(path)
    plt.imsave('mask.png', data_reader.y[0].reshape(256, 256))
    plt.imsave('img.png', data_reader.x[0].reshape(256, 256))
    device = torch.device('cuda')
    dataset = VentricleSegmentationDataset(data_reader.x, data_reader.y, device)
    loader_train_accuracy = DataLoader(dataset, 1)
    print('Data len: ', len(dataset))
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1)),
        nn.BatchNorm2d(3),
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )

    state_d = torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "trained_model.pth"))
    model.load_state_dict(state_d)
    model.to(device)
    model.eval()

    print("Dice: ", calc_dice(model, loader_train_accuracy))


if __name__ == '__main__':
    run_train()
    #    run_train_on_pretrained()
    # eval()
    # warp()

