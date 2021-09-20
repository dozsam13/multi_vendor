from data_processing.adversarial.data_reader import GanDataReader
from torch.utils.data import DataLoader
import sys
import torch
from data_processing.adversarial.ventricle_segmentation_dataset import GanVentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import util
import os
import pathlib
import numpy as np
from data_processing import etlstream


def plot_data(data_with_label, filename):
    plt.clf()
    for (data, label) in data_with_label:
        plt.plot(data, label=label)
    plt.legend()
    plt.savefig(filename)


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


def eval(eval_sources):
    device = torch.device('cuda')
    path = sys.argv[1]
    data_reader = GanDataReader(path, eval_sources)
    print(len(data_reader.x))
    dataset = GanVentricleSegmentationDataset(data_reader.x, data_reader.y, data_reader.vendor, device)
    loader_train_accuracy = DataLoader(dataset, 1)

    model = nn.Sequential(
        pre_model,
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )

    state_d = torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "adversial_trained_model.pth"))
    model.load_state_dict(state_d)
    model.to(device)
    model.eval()

    print("Dice: ", calc_dice(model, loader_train_accuracy))


def train(sources):
    path = sys.argv[1]
    data_reader = GanDataReader(path, sources)

    (x_train, y_train, vendor_train), (x_test, y_test, vendor_test), (_, _, _) = split_data(0.66, 0.99, data_reader.x,
                                                                                            data_reader.y,
                                                                                            data_reader.vendor)
    print(len(x_train), len(x_test))
    batch_size = 15
    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.3, 0.3)),
        transforms.ToTensor()
    ])
    device = torch.device('cuda')
    dataset_train = GanVentricleSegmentationDataset(x_train, y_train, vendor_train, device, augmenter)
    loader_train = DataLoader(dataset_train, batch_size)
    loader_train_accuracy = DataLoader(dataset_train, 1)

    dataset_dev = GanVentricleSegmentationDataset(x_test, y_test, vendor_test, device)
    loader_dev = DataLoader(dataset_dev, batch_size)
    loader_dev_accuracy = DataLoader(dataset_dev, 1)

    model = nn.Sequential(
        pre_model,
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
    s_optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0)
    d_optimizer = optim.AdamW(d_part.parameters(), lr=0.0005, weight_decay=0.1)
    p_optimizer = optim.AdamW(pre_model.parameters(), lr=0.003, weight_decay=0.1)
    s_train_losses = []
    s_dev_losses = []
    d_train_losses = []

    epochs = 10
    for epoch in range(epochs):
        s_train_loss = 0.0
        d_train_loss = 0.0
        for index, sample in enumerate(loader_train):
            img = sample['image']
            target_mask = sample['target']
            target_vendor = sample['vendor']

            if epoch < 5 or epoch > 6:
                predicted_mask = model(img)
                s_loss = s_criterion(predicted_mask, target_mask)
                s_optimizer.zero_grad()
                s_loss.backward()
                s_optimizer.step()
                s_train_loss += s_loss.cpu().detach().numpy()

            if epoch > 4 and epoch < 7:
                predicted_vendor = discriminator(img)
                d_loss = d_criterion(predicted_vendor, target_vendor)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_train_loss += d_loss.cpu().detach().numpy()

                predicted_vendor = discriminator(img)
                p_loss = -1 * d_criterion(predicted_vendor, target_vendor)
                p_optimizer.zero_grad()
                p_loss.backward()
                p_optimizer.step()

        d_train_losses.append(d_train_loss)
        s_train_losses.append(s_train_loss)
        s_dev_losses.append(calculate_loss(loader_dev, model, s_criterion))
        util.progress_bar(epoch + 1, epochs, 50, prefix='Training:')
    plot_data([(s_train_losses, 'train_losses'), (s_dev_losses, 'dev_losses'), (d_train_losses, 'discriminator')],
              'losses.png')
    print("Train dice: ", calc_dice(model, loader_train_accuracy))
    print("Test dice: ", calc_dice(model, loader_dev_accuracy))

    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "adversial_trained_model.pth")
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    train_sources = [etlstream.Origin.SB, etlstream.Origin.ST11]
    eval_source = [etlstream.Origin.MC7]

    train(train_sources)
    eval(eval_source)