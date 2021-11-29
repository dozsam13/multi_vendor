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
from augmentation.domain_augmentation import DomainAugmentation
import json


class Discriminator(nn.Module):
    def __init__(self, channel_n, vendor_n):
        super(Discriminator, self).__init__()
        # self.net = nn.Sequential(
        #     self._block(1, channel_n, 4, 2, 1),
        #     self._block(channel_n, channel_n * 2, 4, 2, 1),
        #     self._block(channel_n * 2, channel_n * 4, 4, 2, 1),
        #     self._block(channel_n * 4, channel_n * 8, 4, 2, 1),
        #     self._block(channel_n * 8, channel_n * 12, 4, 2, 1),
        #     self._block(channel_n * 12, channel_n * 15, 4, 2, 1),
        #     nn.Linear(vendor_n, 480)
        # )
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=(2, 2), stride=1)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=13, kernel_size=(3, 3), stride=1)
        self.linear = nn.Linear(468, vendor_n)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = nn.ReLU()

    # def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    #         nn.LeakyReLU(0.2)
    #     )

    def forward(self, x):
        # s = self.net(x)
        temp = self.maxpool_2_2(self.relu(self.conv1(x)))
        temp = self.maxpool_2_2(self.relu(self.conv2(temp)))
        temp = self.maxpool_2_2(self.relu(self.conv3(temp)))
        temp = self.maxpool_2_2(self.relu(self.conv4(temp)))
        temp = self.maxpool_2_2(self.relu(self.conv5(temp)))
        temp = temp.view(-1, 468)

        return self.linear(temp)


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


pre_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=2),
    nn.BatchNorm2d(3),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), padding=0),
    nn.BatchNorm2d(3),
    nn.ReLU(),
    nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), padding=0),
    nn.BatchNorm2d(3),
    nn.ReLU(),
)

def train(train_sources, eval_source):
    path = sys.argv[1]
    data_reader = MultiSourceDataReader(path, train_sources)
    print(len(data_reader.x))

    (x_train, y_train, vendor_train), (x_dev, y_dev, vendor_dev), (_, _, _) = split_data(0.7, 0.9999, data_reader.x,
                                                                                         data_reader.y,
                                                                                         data_reader.vendor)
    print(len(x_train), len(x_dev))
    batch_size = 8
    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.0, 0.3), scale=(0.8, 1.2)),
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
    (x_eval_domain, y_eval_domain), _, _ = split_data_single_source(0.99, 0.999, datareader_eval_domain.x, datareader_eval_domain.y)
    dataset_eval_domain = VentricleSegmentationDataset(x_eval_domain, y_eval_domain, device)
    loader_eval_domain = DataLoader(dataset_eval_domain, batch_size)
    loader_eval_accuracy = DataLoader(dataset_eval_domain, 1)

    segmentator = nn.Sequential(
        pre_model,
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )

    discriminator = nn.Sequential(
        pre_model,
        Discriminator(2, len(train_sources))
    )

    discriminator.to(device)
    segmentator.to(device)

    s_criterion = nn.BCELoss()
    d_criterion = nn.CrossEntropyLoss()
    s_optimizer = optim.AdamW(segmentator.parameters(), lr=0.00001, weight_decay=0.2)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.0001, weight_decay=0.01)
    a_optimizer = optim.AdamW(pre_model.parameters(), lr=0.001, weight_decay=0.01)
    lmbd = 1.0
    s_train_losses = []
    s_dev_losses = []
    d_train_losses = []
    eval_domain_losses = []
    train_dices = []
    dev_dices = []
    eval_dices = []
    start_time = datetime.now()
    epochs = 300
    calc_dices = True
    for epoch in range(epochs):
        if 20 < epoch < 60:
            lmbd += 0.00001
        if 200 < epoch < 240:
            lmbd -= 0.00001
        s_train_loss = 0.0
        d_train_loss = 0.0
        for index, sample in enumerate(loader_s_train):
            img = sample['image']
            target_mask = sample['target']
            target_vendor = sample['vendor']

            if True: #epoch < 40 or epoch > 70:
                # segmentator
                predicted_mask = segmentator(img)
                s_loss = lmbd*s_criterion(predicted_mask, target_mask)
                s_optimizer.zero_grad()
                s_loss.backward()
                s_optimizer.step()
                s_train_loss += s_loss.cpu().detach().numpy()

            if False:#epoch >= 40:
                # discriminator
                predicted_vendor = discriminator(img)
                d_loss = d_criterion(predicted_vendor, target_vendor)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_train_loss += d_loss.cpu().detach().numpy()

            if False: #epoch > 70:
                # adversarial
                predicted_vendor = discriminator(img)
                a_loss = -1 * lmbd * d_criterion(predicted_vendor, target_vendor)
                a_optimizer.zero_grad()
                a_loss.backward()
                a_optimizer.step()
                lmbd += 1/150
        ###########################################
        d_train_losses.append(d_train_loss / len(loader_s_train))
        s_train_losses.append(s_train_loss / len(loader_s_train))
        s_dev_losses.append(calculate_loss(loader_s_dev, segmentator, s_criterion))
        eval_domain_losses.append(calculate_loss(loader_eval_domain, segmentator, s_criterion))
        if calc_dices and epoch % 3 == 0:
           train_dices.append(calculate_dice(segmentator, loader_s_train_accuracy))
           dev_dices.append(calculate_dice(segmentator, loader_s_dev_accuracy))
           eval_dices.append(calculate_dice(segmentator, loader_eval_accuracy))
        segmentator.train()
        util.progress_bar_with_time(epoch + 1, epochs, start_time)
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model", "weights", "premodel_unet","pretrained_segmentator_unetv0_"+str(date_time)+".pth")
    torch.save(segmentator.state_dict(), model_path)

    util.plot_data_list([(s_train_losses, 'train_losses'), (s_dev_losses, 'dev_losses')],
              'losses' + str(date_time)+'.png')
    util.plot_dice([(train_dices, 'train_dice'), (dev_dices, 'dev_dice')],
              'dices' + str(date_time) + '.png')

    dice_dump_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model", "weights", "premodel_unet",
                              "diceDump" + str(date_time) + ".txt")
    dice_file = open(dice_dump_path, "w+")
    dice_file.write(json.dumps([train_dices, dev_dices, eval_dices]))
    dice_file.close()
    loss_dump_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model", "weights", "premodel_unet",
                                  "lossDump" + str(date_time) + ".txt")
    loss_file = open(loss_dump_path, "w+")
    loss_file.write(json.dumps([s_train_losses, s_dev_losses, d_train_losses]))
    loss_file.close()
    print(max(train_dices), max(dev_dices), max(eval_dices))

    return segmentator


def check_perf(train_sources, eval_source):
    path = sys.argv[1]
    data_reader = MultiSourceDataReader(path, train_sources)
    print(len(data_reader.x))

    (x_train, y_train, vendor_train), (x_dev, y_dev, vendor_dev), (_, _, _) = split_data(0.7, 0.9999, data_reader.x,
                                                                                         data_reader.y,
                                                                                         data_reader.vendor)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    batch_size = 8
    dataset_s_train = MultiSourceDataset(x_train, y_train, vendor_train, device, DomainAugmentation())
    loader_s_train_accuracy = DataLoader(dataset_s_train, 1)

    dataset_s_dev = MultiSourceDataset(x_dev, y_dev, vendor_dev, device)
    loader_s_dev = DataLoader(dataset_s_dev, batch_size)
    loader_s_dev_accuracy = DataLoader(dataset_s_dev, 1)

    datareader_eval_domain = DataReader(os.path.join(path, MultiSourceDataReader.vendors[eval_source]))
    (x_eval_domain, y_eval_domain), _, _ = split_data_single_source(0.99, 0.999, datareader_eval_domain.x,
                                                                              datareader_eval_domain.y)
    dataset_eval_domain = VentricleSegmentationDataset(x_eval_domain, y_eval_domain, device)
    loader_eval_domain = DataLoader(dataset_eval_domain, batch_size)
    loader_eval_accuracy = DataLoader(dataset_eval_domain, 1)

    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model", "weights", "pretrained_segmentator.pth")

    segmentator = UNet()
    segmentator.load_state_dict(torch.load(model_path))

    sigmoid = nn.Sigmoid()
    selector = Selector()
    eval_model = nn.Sequential(segmentator, selector, sigmoid)
    eval_model.to(device)
    eval_model.eval()

    print(calculate_dice(eval_model, loader_s_train_accuracy))
    print(calculate_dice(eval_model, loader_s_dev_accuracy))
    print(calculate_dice(eval_model, loader_eval_accuracy))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()

    train_sources = [etlstream.Origin.ST11, etlstream.Origin.MC7]
    eval_source = etlstream.Origin.SB
    train(train_sources, eval_source)

    # train_sources = [etlstream.Origin.MC7, etlstream.Origin.SB]
    # eval_source = etlstream.Origin.ST11
    # train(train_sources, eval_source)
    #
    # train_sources = [etlstream.Origin.ST11, etlstream.Origin.SB]
    # eval_source = etlstream.Origin.MC7
    # train(train_sources, eval_source)
    #check_perf(train_sources, eval_source)