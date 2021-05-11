from gan.data_reader import GanDataReader
from torch.utils.data import DataLoader
import sys
import torch
from gan.ventricle_segmentation_dataset import GanVentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import utils
import os
import pathlib
import img_warp
import numpy as np
import etlstream


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


def split_data(ratio1, ratio2, data_x, data_y, data_vendor):
    n = len(data_x)
    test_x = data_x[int(n * ratio2):]
    test_y = data_y[int(n * ratio2):]
    test_vendor = data_vendor[int(n * ratio2):]
    data_dev_train_x = data_x[:int(n * ratio2)]
    data_dev_train_y = data_y[:int(n * ratio2)]
    data_dev_train_vendor = data_vendor[:int(n * ratio2)]
    n = len(data_dev_train_x)
    indices = list(range(len(data_dev_train_x)))
    np.random.shuffle(indices)
    train_indices = indices[:int(n * ratio1)]
    dev_indices = indices[int(n * ratio1):int(n * ratio2)]
    train_x = [data_dev_train_x[idx] for idx in train_indices]
    train_y = [data_dev_train_y[idx] for idx in train_indices]
    train_vendor = [data_dev_train_vendor[idx] for idx in train_indices]
    dev_x = [data_dev_train_x[idx] for idx in dev_indices]
    dev_y = [data_dev_train_y[idx] for idx in dev_indices]
    dev_vendor = [data_dev_train_vendor[idx] for idx in dev_indices]

    return (train_x, train_y, train_vendor), (dev_x, dev_y, dev_vendor), (test_x, test_y, test_vendor)


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


def t():
    device = torch.device('cuda')
    discriminator = Discriminator(2, 2)
    discriminator.to(device)
    image_ = np.random.randn(3, 1, 256, 256)
    img_tensor = torch.tensor(image_, dtype=torch.float, device=device)
    r = discriminator(img_tensor)
    print(r.shape)


def train():
    sources = [etlstream.Origin.SB, etlstream.Origin.MC7]
    path = sys.argv[1]
    data_reader = GanDataReader(path, sources)

    (x_train, y_train, vendor_train), (x_test, y_test, vendor_test), (_, _, _) = split_data(0.1, 0.2, data_reader.x, data_reader.y, data_reader.vendor)
    batch_size = 15
    augmenter = transforms.Compose([
        img_warp.SineWarp(10),
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
        transforms.ToTensor()
    ])
    device = torch.device('cuda')
    dataset_train = GanVentricleSegmentationDataset(x_train, y_train, vendor_train, device, augmenter)
    loader_train = DataLoader(dataset_train, batch_size)
    loader_train_accuracy = DataLoader(dataset_train, 1)

    dataset_dev = GanVentricleSegmentationDataset(x_test, y_test, vendor_test, device)
    loader_dev_accuracy = DataLoader(dataset_dev, 1)
    pre_model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=2),
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), padding=0),
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), padding=0),
    )

    model = nn.Sequential(
        pre_model,
        nn.BatchNorm2d(3),
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )

    d_part = Discriminator(2, len(sources))
    discriminator = nn.Sequential(
        pre_model,
        d_part
    )
    discriminator.to(device)
    model.to(device)
    s_criterion = nn.BCELoss()
    d_criterion = nn.CrossEntropyLoss()
    s_optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    d_optimizer = optim.AdamW(d_part.parameters(), lr=0.0005, weight_decay=0.1)
    p_optimizer = optim.AdamW(pre_model.parameters(), lr=0.0005, weight_decay=0.1)
    s_train_losses = []
    d_train_losses = []

    epochs = 1
    for epoch in range(epochs):
        s_train_loss = 0.0
        d_train_loss = 0.0
        for index, sample in enumerate(loader_train):
            img = sample['image']
            target_mask = sample['target']
            target_vendor = sample['vendor']

            predicted_mask = model(img)
            s_loss = s_criterion(predicted_mask, target_mask)
            s_optimizer.zero_grad()
            s_loss.backward()
            s_optimizer.step()
            s_train_loss += s_loss.cpu().detach().numpy()

            if epoch % 100 == 0:
                predicted_vendor = discriminator(img)
                d_loss = d_criterion(predicted_vendor, target_vendor)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_train_loss += d_loss.cpu().detach().numpy()

                predicted_vendor = discriminator(img)
                p_loss = -1*d_criterion(predicted_vendor, target_vendor)
                p_optimizer.zero_grad()
                p_loss.backward()
                p_optimizer.step()

                d_train_losses.append(d_train_loss)
            s_train_losses.append(s_train_loss)
    print("Train dice: ", calc_dice(model, loader_train_accuracy))
    print("Test dice: ", calc_dice(model, loader_dev_accuracy))



    #     train_losses.append(train_loss / len(loader_train))
    #     dev_losses.append(calculate_loss(loader_dev, model, criterion))
    #     utils.progress_bar(epoch + 1, epochs, 50, prefix='Training:')
    # plot_data(train_losses, 'train_losses', dev_losses, 'dev_losses', 'losses.png')
    # model.eval()
    # model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "trained_model.pth")
    # torch.save(model.state_dict(), model_path)
    # print("Train dice: ", calc_dice(model, loader_train_accuracy))
    # print("Test dice: ", calc_dice(model, loader_dev_accuracy))
    # pred_mask = torch.round(model(dataset_dev[0]['image'].unsqueeze(0))).cpu().detach().numpy().reshape(256, 256)
    # expected_mask = dataset_dev[0]['target'].cpu().detach().numpy().reshape(256, 256)
    # plt.imsave('mask.png', pred_mask)
    # plt.imsave('mask_expected.png', expected_mask)
    # plt.imsave('image.png', dataset_dev[0]['image'][0, :, :].cpu().detach().numpy().reshape(256, 256))


if __name__ == '__main__':
    #t()
    train()