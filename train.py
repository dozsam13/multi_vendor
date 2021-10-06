from data_processing.normal.data_reader import DataReader
from torch.utils.data import DataLoader
import sys
import torch
from data_processing.normal.ventricle_segmentation_dataset import VentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import util
import os
import pathlib
from model.unet import UNet
from train_util import *
import random
import gc
from datetime import datetime
from train_util import Selector


def split_data(ratio1, ratio2, data_x, data_y):
    indices = [*range(len(data_x))]
    n = len(data_x)
    np.random.shuffle(indices)

    train_indices = indices[:int(n * ratio1)]
    dev_indices = indices[int(n * ratio1):int(n * ratio2)]
    test_indices = indices[int(n * ratio2):]

    train_x = flatten([data_x[idx] for idx in train_indices])
    train_y = flatten([data_y[idx] for idx in train_indices])
    dev_x = flatten([data_x[idx] for idx in dev_indices])
    dev_y = flatten([data_y[idx] for idx in dev_indices])
    test_x = flatten([data_x[idx] for idx in test_indices])
    test_y = flatten([data_y[idx] for idx in test_indices])

    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)


def run_train(run_on_pretrained, path):
    data_reader = DataReader(path)

    (x_train, y_train), (x_test, y_test), (_, _) = split_data(0.05, 0.1, data_reader.x, data_reader.y)
    print("X_train: ", len(x_train), "X_dev: ", len(x_test))
    batch_size = 5
    augmenter = transforms.Compose([
        #img_warp.SineWarp(10),
        #warp2.ElasticTransform(),
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
        transforms.ToTensor()
    ])
    device = torch.device('cpu')
    dataset_train = VentricleSegmentationDataset(x_train, y_train, device, augmenter)
    loader_train = DataLoader(dataset_train, batch_size)
    loader_train_accuracy = DataLoader(dataset_train, 1)

    dataset_dev = VentricleSegmentationDataset(x_test, y_test, device)
    loader_dev = DataLoader(dataset_dev, batch_size)
    loader_dev_accuracy = DataLoader(dataset_dev, 1)

    model = nn.Sequential(UNet(), Selector(), nn.Sigmoid())

    if run_on_pretrained:
        state_d = torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_model.pth"))
        model.load_state_dict(state_d)
    model.eval()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    epochs = 90
    train_losses = []
    dev_losses = []
    train_dices = []
    dev_dices = []
    calc_dice = False
    start_time = datetime.now()
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
        model.eval()
        if calc_dice and epoch % 3 == 0:
            model.eval()
            train_dices.append(calculate_dice(model, loader_train_accuracy))
            dev_dices.append(calculate_dice(model, loader_dev_accuracy))
        train_losses.append(train_loss / len(loader_train))
        dev_losses.append(calculate_loss(loader_dev, model, criterion))
        util.progress_bar_with_time(epoch + 1, epochs, start_time)
        model.train()
    util.plot_data(train_losses, 'train_losses', dev_losses, 'dev_losses', 'losses.png')
    util.plot_data(train_dices, 'train_dices', dev_dices, 'dev_dices', 'dices.png')

    if not run_on_pretrained:
        model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_model.pth")
        torch.save(model.state_dict(), model_path)

    model.eval()
    print("Train dice: ", calculate_dice(model, loader_train_accuracy))
    print("Test dice: ", calculate_dice(model, loader_dev_accuracy))
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

    print("Dice: ", calculate_dice(model, loader_train_accuracy))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()
    run_train(run_on_pretrained=False, path=sys.argv[1])
