import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_loss(dataset, model, criterion, batch_size):
    loader = DataLoader(dataset, batch_size)
    loss_sum = 0.0
    for sample in loader:
        image = sample['image']
        target = sample['target']

        predicted = model(image)

        loss = criterion(predicted, target)
        loss_sum += loss.cpu().detach().numpy()
        del loss

    return loss_sum / len(loader)


def calculate_dice(model, dataset):
    data_loader = DataLoader(dataset, 1)
    s = 0
    for sample in data_loader:
        img = sample['image']
        target = sample['target'].cpu().detach().numpy()
        predicted = model(img)
        predicted = torch.round(predicted).cpu().detach().numpy()
        s += calc_dice_for_img(predicted, target)
    return s / len(data_loader)


def calculate_dice_values(model, data_loader):
    dices = []
    for sample in data_loader:
        img = sample['image']
        target = sample['target'].cpu().detach().numpy()
        predicted = model(img)
        predicted = torch.round(predicted).cpu().detach().numpy()
        dices.append(calc_dice_for_img(predicted, target))
    return dices


def calc_dice_for_img(predicted, target):
    smooth = 1.

    pred_flat = predicted.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = np.dot(pred_flat, target_flat)

    return (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)


def flatten(t):
    return [item for sublist in t for item in sublist]


class Selector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input[0]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice