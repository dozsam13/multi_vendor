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

def run_train(path,path2, net_name, rate):
    data_reader = DataReader(path)
    (x_train, y_train), (x_test, y_test), (_, _) = split_data(0.75, 0.99, data_reader.x, data_reader.y)
    print("X_train: ", len(x_train), "X_dev: ", len(x_test))
    new_domain_data_reader = DataReader(path2)
    (x_train_new, y_train_new), (_, _), (x_test_new, y_test_new) = split_data(0.75*rate, 0.85, new_domain_data_reader.x, new_domain_data_reader.y)
    print("X_train: ", len(x_train_new), "X_dev: ", len(x_test_new))
    print("új aránya ", float(len(x_train_new))/float(len(x_train)))
    x_train = x_train_new
    y_train = y_train_new
    #l = list(zip(x_train, y_train))
    #np.random.shuffle(l)

    #[x_train, y_train] = zip(*l)

    batch_size = 16
    augmenter = transforms.Compose([
        #img_warp.SineWarp(10),
        #warp2.ElasticTransform(),
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45], translate=(0.0, 0.3), scale=(0.8, 1.2)),
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

    dataset_dev_new = VentricleSegmentationDataset(x_test_new, y_test_new, device)
    loader_dev_new = DataLoader(dataset_dev_new, batch_size)
    loader_dev_accuracy_new = DataLoader(dataset_dev_new, 1)

    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1)),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True),
        )
    model.train()

    model.to(device)
    model.train()

    criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.2)
    epochs = 250
    train_losses = []
    dev_losses = []
    dev_new_losses = []
    start_time = datetime.now()
    train_dices = []
    dev_dices = []
    dev_new_dices = []
    calc_dices = True
    lmbd = 1.0
    for epoch in range(epochs):
        if epoch > 20 and epoch < 60:
            lmbd += 0.00001
        if epoch > 200 and epoch < 240:
            lmbd -= 0.00001
        train_loss = 0.0
        model.train()
        for index, sample in enumerate(loader_train):
            img = sample['image']
            target = sample['target']
            predicted = model(img)
            loss = lmbd*criterion(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().numpy()
        model.eval()
        train_losses.append(train_loss / len(loader_train))
        dev_losses.append(calculate_loss(loader_dev, model, criterion))
        dev_new_losses.append(calculate_loss(loader_dev_new, model, criterion))
        util.progress_bar_with_time(epoch + 1, epochs, start_time)
        if calc_dices and epoch % 3 == 0:
            train_dices.append(calculate_dice(model, loader_train_accuracy))
            dev_dices.append(calculate_dice(model, loader_dev_accuracy))
            dev_new_dices.append(calculate_dice(model, loader_dev_accuracy_new))
    date_time = str(datetime.now().strftime("%m%d%Y_%H%M%S"))
    util.plot_data_list([(train_losses, 'train_losses'), (dev_losses, 'dev_losses'), (dev_new_losses, 'st11_dev_losses')], 'losses'+date_time+'.png')
    util.plot_dice([(train_dices, 'train_dice'), (dev_dices, 'dev_dice'), (dev_new_dices, 'st11_dev_dice')], 'dices'+date_time+'.png')
    #util.boxplot([dev_dices, dev_new_dices], ['SB', 'ST11', 'MC7'], 'new_domain_boxplot'+date_time+'.png')
    #model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), net_name)
    #torch.save(model.state_dict(), model_path)
    print("max train-dev dices", max(train_dices), " ", max(dev_dices), " ", max(dev_new_dices))
    print('ez volt a 20 mintas')
    #return calculate_dice_values(model, loader_dev_accuracy)


def check_dice(path):
    data_reader = DataReader(path)
    (x_train, y_train), (_, _), (_, _) = split_data(0.99, 0.99999, data_reader.x, data_reader.y)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    dataset_dev = VentricleSegmentationDataset(x_train, y_train, device)
    loader_dev_accuracy = DataLoader(dataset_dev, 1)

    adv_model = nn.Sequential(
        pre_model,
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )
    multi_model = nn.Sequential(
        pre_model,
        torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                       init_features=32, pretrained=True)
    )

    state_d = torch.load('/userhome/student/dozsa/multi_vendor/model/weights/premodel_unet/adv_trained_segmentator_12042021_001052.pth')
    adv_model.load_state_dict(state_d)
    adv_model.to(device)

    state_d = torch.load('/userhome/student/dozsa/multi_vendor/model/weights/premodel_unet/basic_multitrain/SBST11_pretrained_segmentator_unetv0_11302021_180842.pth')
    multi_model.load_state_dict(state_d)
    multi_model.to(device)


    #print(calculate_dice(model, loader_dev_accuracy))

    #return calculate_dice_values(model, loader_dev_accuracy)

    
    expected_mask = dataset_dev[0]['target'].cpu().detach().numpy()

    for data in dataset_dev:
        adv_mask = torch.round(adv_model(data['image'].unsqueeze(0)))
        multi_mask = torch.round(multi_model(data['image'].unsqueeze(0)))
        expected_mask = data['target'].cpu().detach().numpy()

        adv_dice = calc_dice_for_img(adv_mask.cpu().detach().numpy(), data['target'].cpu().detach().numpy())
        multi_dice = calc_dice_for_img(multi_mask.cpu().detach().numpy(), data['target'].cpu().detach().numpy())

        if adv_dice < multi_dice-0.2 and multi_dice > 0.8:
            print(adv_dice, multi_dice)
            plt.subplot(1, 3, 1)
            plt.imshow(expected_mask.reshape(256, 256), cmap='gray')
            plt.axis('off')
            plt.title("a")
            plt.subplot(1, 3, 2)
            plt.imshow(multi_mask.cpu().detach().numpy().reshape(256, 256), cmap='gray')
            plt.axis('off')
            plt.title("b")
            plt.subplot(1, 3, 3)
            plt.imshow(adv_mask.cpu().detach().numpy().reshape(256, 256), cmap='gray')
            plt.axis('off')
            plt.title("c")
            
            plt.savefig('masks3.png')
            exit()        
    
    #plt.imsave('mask.png', pred_mask, cmap='gray')
    #plt.imsave('mask_expected.png', expected_mask, cmap='gray')
    #plt.imsave('image.png', dataset_dev[0]['image'][0, :, :].cpu().detach().numpy().reshape(256, 256), cmap='gray')

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()
    #run_train(path='/userhome/student/dozsa/multi_vendor/out_filled/MC7', path2='/userhome/student/dozsa/multi_vendor/out_filled/ST11', net_name="sb_normal_train.pth", rate=0.05)
    run_train(path='/userhome/student/dozsa/multi_vendor/out_filled/MC7', path2='/userhome/student/dozsa/multi_vendor/out_filled/ST11', net_name="sb_normal_train.pth", rate=0.40)
    #dices2 = check_dice(path='/userhome/student/dozsa/multi_vendor/out_filled/MC7', net_name="sb_normal_train.pth")
    #dices3 = check_dice(path='/userhome/student/dozsa/multi_vendor/out_filled/ST11', net_name="sb_normal_train.pth")
    #util.boxplot([sb_dices, dices3, dices2], ['SB', 'ST11', 'MC7'], 'sb_idegen_domain_boxplot.png')

    #st_dices = run_train(path='/userhome/student/dozsa/multi_vendor/out_filled/ST11')
    #mc_dices = run_train(path='/userhome/student/dozsa/multi_vendor/out_filled/MC7')
    #util.boxplot([sb_dices, st_dices, mc_dices], ['SB', 'ST11', 'MC7'], 'sb_boxplot.png')
    #check_dice(path='/userhome/student/dozsa/multi_vendor/out_filled/MC7')

