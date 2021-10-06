from data_processing.adversarial.data_reader import MultiSourceDataReader
from torch.utils.data import DataLoader
import sys
import torch
from data_processing.adversarial.ventricle_segmentation_dataset import MultiSourceDataset
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


def plot_data(data_with_label, filename):
    plt.clf()
    for (data, label) in data_with_label:
        plt.plot(data, label=label)
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


def eval(eval_sources):
    device = torch.device('cuda')
    path = sys.argv[1]
    data_reader = MultiSourceDataReader(path, eval_sources)
    print(len(data_reader.x))
    dataset = MultiSourceDataset(data_reader.x, data_reader.y, data_reader.vendor, device)
    loader_train_accuracy = DataLoader(dataset, 1)

    model = nn.Sequential(UNet(), Selector(), nn.Sigmoid())

    state_d = torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "adversial_trained_model.pth"))
    model.load_state_dict(state_d)
    model.to(device)
    model.eval()

    print("Dice: ", calculate_dice(model, loader_train_accuracy))


def train(sources):
    path = sys.argv[1]
    data_reader = MultiSourceDataReader(path, sources)

    (x_train, y_train, vendor_train), (x_dev, y_dev, vendor_dev), (_, _, _) = split_data(0.1, 0.2, data_reader.x,
                                                                                            data_reader.y,
                                                                                            data_reader.vendor)
    print(len(x_train), len(x_dev))
    batch_size = 5
    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
        transforms.ToTensor()
    ])
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    dataset_train = MultiSourceDataset(x_train, y_train, vendor_train, device, augmenter)
    loader_train = DataLoader(dataset_train, batch_size)
    loader_train_accuracy = DataLoader(dataset_train, 1)

    dataset_dev = MultiSourceDataset(x_dev, y_dev, vendor_dev, device)
    loader_dev = DataLoader(dataset_dev, batch_size)
    loader_dev_accuracy = DataLoader(dataset_dev, 1)

    segmentator = UNet()

    discriminator = Discriminator(n_domains=len(sources))

    sigmoid = nn.Sigmoid()
    selector = Selector()

    discriminator.to(device)
    segmentator.to(device)
    s_criterion = nn.BCELoss()
    d_criterion = nn.CrossEntropyLoss()
    s_optimizer = optim.AdamW(segmentator.parameters(), lr=0.0005, weight_decay=0.0)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.0005, weight_decay=0.1)
    a_optimizer = optim.AdamW(segmentator.encoder.parameters(), lr=0.003, weight_decay=0.1)
    s_train_losses = []
    s_dev_losses = []
    d_train_losses = []
    start = datetime.now()
    epochs = 1
    for epoch in range(epochs):
        s_train_loss = 0.0
        d_train_loss = 0.0
        for index, sample in enumerate(loader_train):
            img = sample['image']
            target_mask = sample['target']
            target_vendor = sample['vendor']

            # segmentator
            predicted_mask, inner_repr = segmentator(img)
            predicted_mask = sigmoid(predicted_mask)
            s_loss = s_criterion(predicted_mask, target_mask)
            s_optimizer.zero_grad()
            s_loss.backward()
            s_optimizer.step()
            s_train_loss += s_loss.cpu().detach().numpy()

            # discriminator
            predicted_mask = predicted_mask.cpu().detach()
            inner_repr = inner_repr.cpu().detach()
            predicted_mask.to(device)
            inner_repr.to(device)
            predicted_vendor = discriminator(predicted_mask, inner_repr)
            d_loss = d_criterion(predicted_vendor, target_vendor)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            d_train_loss += d_loss.cpu().detach().numpy()

            # adversarial
            predicted_mask, inner_repr = segmentator(img)
            predicted_vendor = discriminator(predicted_mask, inner_repr)
            a_loss = -1 * d_criterion(predicted_vendor, target_vendor)
            a_optimizer.zero_grad()
            a_loss.backward()
            a_optimizer.step()

        ###########################################
        d_train_losses.append(d_train_loss)
        s_train_losses.append(s_train_loss)
        s_dev_losses.append(calculate_loss(loader_dev, nn.Sequential(segmentator, selector, sigmoid), s_criterion))
        util.progress_bar_with_time(epoch + 1, epochs, start_time)
    plot_data([(s_train_losses, 'train_losses'), (s_dev_losses, 'dev_losses'), (d_train_losses, 'discriminator_losses')],
              'losses.png')
    print("Train dice: ", calculate_dice(nn.Sequential(segmentator, selector, sigmoid), loader_train_accuracy))
    print("Test dice: ", calculate_dice(nn.Sequential(segmentator, selector, sigmoid), loader_dev_accuracy))

    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "adversial_trained_model.pth")
    torch.save(segmentator.state_dict(), model_path)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()

    train_sources = [etlstream.Origin.SB, etlstream.Origin.MC7]
    eval_source = [etlstream.Origin.ST11]

    train(train_sources)
    #eval(eval_source)