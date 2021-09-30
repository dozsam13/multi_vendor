import torch
import numpy as np


def calculate_loss(loader, model, criterion):
    loss_sum = 0.0
    for sample in loader:
        image = sample['image']
        target = sample['target']

        predicted = model(image)[0]

        loss = criterion(predicted, target)
        loss_sum += loss.cpu().detach().numpy()
        del loss

    return loss_sum / len(loader)


def calculate_dice(model, data_loader):
    s = 0
    for sample in data_loader:
        img = sample['image']
        target = sample['target'].cpu().detach().numpy()
        predicted = model(img)[0]
        predicted = torch.round(predicted).cpu().detach().numpy()
        s += calc_dice_for_img(predicted, target)
    return s / len(data_loader)


def calc_dice_for_img(predicted, target):
    smooth = 1.

    pred_flat = predicted.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = np.dot(pred_flat, target_flat)

    return (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)


def flatten(t):
    return [item for sublist in t for item in sublist]
