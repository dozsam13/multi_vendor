from data_processing.adversarial.data_reader import MultiSourceDataReader
from data_processing.normal.data_reader import DataReader
from torch.utils.data import DataLoader
import sys
import torch
from data_processing.adversarial.ventricle_segmentation_dataset import MultiSourceDataset
from data_processing.normal.ventricle_segmentation_dataset import VentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import util
import os
import pathlib
import numpy as np
from data_processing import etlstream
from train_util import *
from model.discriminator import Discriminator
from model.unet import UNet
from train_util import Selector
import random
import gc
from datetime import datetime
from train_util import DiceLoss


def plot_data(data_with_label, filename):
    plt.clf()
    for (data, label) in data_with_label:
        plt.plot(data, label=label, linewidth=3.0)
    plt.legend()
    plt.savefig(filename)


def split_data(ratio1, ratio2, data_x, data_y, data_vendor):
    indices = [*range(len(data_x))]
    n = len(data_x)
    np.random.shuffle(indices)

    train_indices = indices[:int(n * ratio1)]
    dev_indices = indices[int(n * ratio1):int(n * ratio2)]
    test_indices = indices[int(n * ratio2):]

    train_x = flatten([data_x[idx] for idx in train_indices])
    train_y = flatten([data_y[idx] for idx in train_indices])
    train_vendor = flatten([data_vendor[idx] for idx in train_indices])
    dev_x = flatten([data_x[idx] for idx in dev_indices])
    dev_y = flatten([data_y[idx] for idx in dev_indices])
    dev_vendor = flatten([data_vendor[idx] for idx in dev_indices])
    test_x = flatten([data_x[idx] for idx in test_indices])
    test_y = flatten([data_y[idx] for idx in test_indices])
    test_vendor = flatten([data_vendor[idx] for idx in test_indices])

    return (train_x, train_y, train_vendor), (dev_x, dev_y, dev_vendor), (test_x, test_y, test_vendor)


def split_data_single_source(ratio1, ratio2, data_x, data_y):
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


def train(train_sources, eval_source, train_ind):
    print('Train index: ', train_ind)
    path = sys.argv[1]
    data_reader = MultiSourceDataReader(path, train_sources)
    print(data_reader.source_dict)

    (x_train, y_train, vendor_train), (x_dev, y_dev, vendor_dev), (_, _, _) = split_data(0.7, 0.9999, data_reader.x,
                                                                                         data_reader.y,
                                                                                         data_reader.vendor)
    print(len(x_train), len(x_dev))
    batch_size = 8
    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
        transforms.ToTensor()
    ])
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    dataset_s_train = MultiSourceDataset(x_train, y_train, vendor_train, device, augmenter)
    loader_s_train = DataLoader(dataset_s_train, batch_size)
    loader_s_train_accuracy = DataLoader(dataset_s_train, 1)

    dataset_s_dev = MultiSourceDataset(x_dev, y_dev, vendor_dev, device)
    loader_s_dev = DataLoader(dataset_s_dev, batch_size)
    loader_s_dev_accuracy = DataLoader(dataset_s_dev, 1)

    datareader_eval_domain = DataReader(os.path.join(path, MultiSourceDataReader.vendors[eval_source]))
    (x_eval_domain, y_eval_domain), (_, _), (_, _) = split_data_single_source(0.99, 0.999, datareader_eval_domain.x,
                                                                              datareader_eval_domain.y)
    dataset_eval_domain = VentricleSegmentationDataset(x_eval_domain, y_eval_domain, device)
    loader_eval_domain = DataLoader(dataset_eval_domain, batch_size)
    loader_eval_accuracy = DataLoader(dataset_eval_domain, 1)

    train_sources.append(eval_source)
    data_reader = MultiSourceDataReader(path, train_sources)
    (x_da_train, y_da_train, vendor_da_train), _, _ = split_data(0.9999, 0.999999, data_reader.x,
                                                                                         data_reader.y,
                                                                                         data_reader.vendor)
    dataset_da_train = MultiSourceDataset(x_da_train, y_da_train, vendor_da_train, device, augmenter)
    loader_da_train = DataLoader(dataset_da_train, batch_size)

    segmentator = UNet()

    discriminator = Discriminator(n_domains=len(train_sources))

    sigmoid = nn.Sigmoid()
    selector = Selector()

    discriminator.to(device)
    segmentator.to(device)
    s_criterion = nn.BCELoss()
    d_criterion = nn.CrossEntropyLoss()
    s_optimizer = optim.AdamW(segmentator.parameters(), lr=0.0005, weight_decay=0.1)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.001, weight_decay=0.2)
    a_optimizer = optim.AdamW(segmentator.encoder.parameters(), lr=0.0005, weight_decay=0.1)
    s_train_losses = []
    s_dev_losses = []
    d_train_losses = []
    eval_domain_losses = []
    train_dices = []
    dev_dices = []
    eval_dices = []
    start_time = datetime.now()
    epochs = 200
    calc_dices = True
    da_loader_iter = iter(loader_da_train)
    for epoch in range(epochs):
        s_train_loss = 0.0
        d_train_loss = 0.0
        for index, sample in enumerate(loader_s_train):
            img = sample['image']
            target_mask = sample['target']

            da_sample = next(da_loader_iter, None)
            if da_sample in None:
                da_loader_iter = iter(loader_da_train)
                da_sample = next(loader_da_train, None)
            if epoch < 40 or epoch > 70:
                # segmentator
                predicted_activations, inner_repr = segmentator(img)
                predicted_mask = sigmoid(predicted_activations)
                s_loss = s_criterion(predicted_mask, target_mask)
                s_optimizer.zero_grad()
                s_loss.backward()
                s_optimizer.step()
                s_train_loss += s_loss.cpu().detach().numpy()

            if epoch >= 40:
                # discriminator
                predicted_activations, inner_repr = segmentator(da_sample['image'])
                predicted_activations = predicted_activations.clone().detach()
                inner_repr = inner_repr.clone().detach()
                predicted_vendor = discriminator(predicted_activations, inner_repr)
                d_loss = d_criterion(predicted_vendor, da_sample['vendor'])
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_train_loss += d_loss.cpu().detach().numpy()

            if epoch > 50:
                # adversarial
                predicted_mask, inner_repr = segmentator(da_sample['image'])
                predicted_vendor = discriminator(predicted_mask, inner_repr)
                a_loss = -1 * d_criterion(predicted_vendor, da_sample['vendor'])
                a_optimizer.zero_grad()
                a_loss.backward()
                a_optimizer.step()
        if epoch == 50:
            model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_segmentator.pth")
            torch.save(segmentator.state_dict(), model_path)
            model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "pretrained_discriminator.pth")
            torch.save(discriminator.state_dict(), model_path)

        ###########################################
        eval_model = nn.Sequential(segmentator, selector, sigmoid)
        eval_model.to(device)
        eval_model.eval()
        d_train_losses.append(d_train_loss / len(loader_s_train))
        s_train_losses.append(s_train_loss / len(loader_s_train))
        s_dev_losses.append(calculate_loss(loader_s_dev, eval_model, s_criterion))
        eval_domain_losses.append(calculate_loss(loader_eval_domain, eval_model, s_criterion))
        if calc_dices and epoch % 3 == 0:
            train_dices.append(calculate_dice(eval_model, loader_s_train_accuracy))
            dev_dices.append(calculate_dice(eval_model, loader_s_dev_accuracy))
            eval_dices.append(calculate_dice(eval_model, loader_eval_accuracy))
        segmentator.train()
        util.progress_bar_with_time(epoch + 1, epochs, start_time)

    plot_data([(s_train_losses, 'train_losses'), (s_dev_losses, 'dev_losses'), (d_train_losses, 'discriminator_losses'),
               (eval_domain_losses, 'eval_domain_losses')],
              'losses' + str(train_ind) + '.png')
    plot_data([(train_dices, 'train_dice'), (dev_dices, 'dev_dice'), (eval_dices, 'eval_dice')],
              'dices' + str(train_ind) + '.png')
    print(max(train_dices), max(dev_dices), max(eval_dices))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()

    train_sources = [etlstream.Origin.ST11, etlstream.Origin.SB]
    eval_source = etlstream.Origin.MC7

    train(train_sources, eval_source, 2)