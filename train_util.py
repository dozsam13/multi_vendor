import torch
import numpy as np
import torch.nn as nn


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


def calculate_dice(model, data_loader):
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
    smooth = 0.000001

    intersection = predicted * target

    return (2. * np.sum(intersection) + smooth) / (np.sum(predicted) + np.sum(target) + smooth)


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