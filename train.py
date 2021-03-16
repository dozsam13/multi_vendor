from data_reader import DataReader
from torch.utils.data import DataLoader
import sys
import torch
from ventricle_segmentation_dataset import VentricleSegmentationDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


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

def run_train():
    path = sys.argv[1]
    data_reader = DataReader(path)

    x_train = data_reader.x[:int(len(data_reader.x)*0.66)]
    y_train = data_reader.y[:int(len(data_reader.x)*0.66)]
    x_test = data_reader.x[int(len(data_reader.x)*0.66):]
    y_test = data_reader.y[int(len(data_reader.x)*0.66):]

    batch_size = 15
    device = torch.device('cuda')
    dataset_train = VentricleSegmentationDataset(x_train, y_train, device)
    loader_train = DataLoader(dataset_train, batch_size)
    dataset_dev = VentricleSegmentationDataset(x_test, y_test, device)
    loader_dev = DataLoader(dataset_dev, batch_size)

    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1)),
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True),
        nn.Sigmoid()
        )

    epochs = 10
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters())
    train_losses = []
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
            break
        train_losses.append(train_loss / len(loader_train))
    print(train_losses)
    plot_data(train_losses, 'train_losses', train_losses, 'train_losses', 'losses.png')

    model.eval()
    print(dataset_train[0]['image'].cpu().detach().numpy().shape)
    pred_mask = model(dataset_train[0]['image']).cpu().detach().numpy()
    target_mask = dataset_train[0]['image'].cpu().detach().numpy()
    plt.imsave('predicted.png', pred_mask )
    plt.imsave('target.png', target_mask )


if __name__ == '__main__':
    run_train()

