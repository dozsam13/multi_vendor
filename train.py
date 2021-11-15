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
from train_util import DiceLoss


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
    (x_train, y_train), (x_test, y_test), (_, _) = split_data(0.65, 0.99, data_reader.x, data_reader.y)
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
    if torch.cuda.is_available():
        device = torch.device('cuda')

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
    epochs = 120
    train_losses = []
    dev_losses = []
    start_time = datetime.now()
    train_dices = []
    dev_dices = []
    calc_dices = True
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
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
        train_losses.append(train_loss / len(loader_train))
        dev_losses.append(calculate_loss(loader_dev, model, criterion))
        util.progress_bar_with_time(epoch + 1, epochs, start_time)
        if calc_dices and epoch % 3 == 0:
            train_dices.append(calculate_dice(model, loader_train_accuracy))
            dev_dices.append(calculate_dice(model, loader_dev_accuracy))
    util.plot_data(train_losses, 'train_losses', dev_losses, 'dev_losses', 'losses.png')
    util.plot_data(train_dices, 'train_dice', dev_dices, 'dev_dice', 'dices.png')
    if not run_on_pretrained:
        model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_model.pth")
        torch.save(model.state_dict(), model_path)

    model.eval()
    print("Max train dice: ", max(train_dices))
    print("Max dev dice: ", max(dev_dices))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()
    run_train(run_on_pretrained=False, path=sys.argv[1])
