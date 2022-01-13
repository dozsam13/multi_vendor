from data_processing.data_reader import DataReader
import sys
from data_processing.dataset import MultiDomainDataset
import torch.optim as optim
import util
import os
import pathlib
from data_processing import etlstream
from train_util import *
from model.discriminator import Discriminator
from model.unet import UNet
from train_util import Selector
import random
import gc
from datetime import datetime
from augmentation.domain_augmentation import DomainAugmentation
from tqdm import tqdm


def train(train_sources, eval_source):
    path = sys.argv[1]
    dr = DataReader(path, train_sources)
    dr.read()
    print(len(dr.train.x))

    batch_size = 8
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    dataset_s_train = MultiDomainDataset(dr.train.x, dr.train.y, dr.train.vendor, device, DomainAugmentation())
    dataset_s_dev = MultiDomainDataset(dr.dev.x, dr.dev.y, dr.dev.vendor, device)
    dataset_s_test = MultiDomainDataset(dr.test.x, dr.test.y, dr.test.vendor, device)
    loader_s_train = DataLoader(dataset_s_train, batch_size, shuffle=True)

    dr_eval = DataReader(path, [eval_source])
    dr_eval.read()

    dataset_eval_dev = MultiDomainDataset(dr_eval.dev.x, dr_eval.dev.y, dr_eval.dev.vendor, device)
    dataset_eval_test = MultiDomainDataset(dr_eval.test.x, dr_eval.test.y, dr_eval.test.vendor, device)

    dataset_da_train = MultiDomainDataset(dr.train.x+dr_eval.train.x, dr.train.y+dr_eval.train.y, dr.train.vendor+dr_eval.train.vendor, device, DomainAugmentation())
    loader_da_train = DataLoader(dataset_da_train, batch_size, shuffle=True)

    segmentator = UNet()
    discriminator = Discriminator(n_domains=len(train_sources))
    discriminator.to(device)
    segmentator.to(device)

    sigmoid = nn.Sigmoid()
    selector = Selector()

    s_criterion = nn.BCELoss()
    d_criterion = nn.CrossEntropyLoss()
    s_optimizer = optim.AdamW(segmentator.parameters(), lr=0.0001, weight_decay=0.01)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.001, weight_decay=0.01)
    a_optimizer = optim.AdamW(segmentator.encoder.parameters(), lr=0.001, weight_decay=0.01)
    lmbd = 1/150
    s_train_losses = []
    s_dev_losses = []
    d_train_losses = []
    eval_domain_losses = []
    train_dices = []
    dev_dices = []
    eval_dices = []
    epochs = 3
    da_loader_iter = iter(loader_da_train)
    for epoch in tqdm(range(epochs)):
        s_train_loss = 0.0
        d_train_loss = 0.0
        for index, sample in enumerate(loader_s_train):
            img = sample['image']
            target_mask = sample['target']

            da_sample = next(da_loader_iter, None)
            if epoch == 100:
                s_optimizer.defaults['lr'] = 0.001
                d_optimizer.defaults['lr'] = 0.0001
            if da_sample is None:
                da_loader_iter = iter(loader_da_train)
                da_sample = next(da_loader_iter, None)
            if epoch < 50 or epoch >= 100:
                # Training step of segmentator
                predicted_activations, inner_repr = segmentator(img)
                predicted_mask = sigmoid(predicted_activations)
                s_loss = s_criterion(predicted_mask, target_mask)
                s_optimizer.zero_grad()
                s_loss.backward()
                s_optimizer.step()
                s_train_loss += s_loss.cpu().detach().numpy()

            if epoch >= 50:
                # Training step of discriminator
                predicted_activations, inner_repr = segmentator(da_sample['image'])
                predicted_activations = predicted_activations.clone().detach()
                inner_repr = inner_repr.clone().detach()
                predicted_vendor = discriminator(predicted_activations, inner_repr)
                d_loss = d_criterion(predicted_vendor, da_sample['vendor'])
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_train_loss += d_loss.cpu().detach().numpy()

            if epoch >= 100:
                # adversarial training step
                predicted_mask, inner_repr = segmentator(da_sample['image'])
                predicted_vendor = discriminator(predicted_mask, inner_repr)
                a_loss = -1 * lmbd * d_criterion(predicted_vendor, da_sample['vendor'])
                a_optimizer.zero_grad()
                a_loss.backward()
                a_optimizer.step()
                lmbd += 1/150
        inference_model = nn.Sequential(segmentator, selector, sigmoid)
        inference_model.to(device)
        inference_model.eval()
        d_train_losses.append(d_train_loss / len(loader_s_train))
        s_train_losses.append(s_train_loss / len(loader_s_train))
        s_dev_losses.append(calculate_loss(dataset_s_dev, inference_model, s_criterion, batch_size))
        eval_domain_losses.append(calculate_loss(dataset_eval_dev, inference_model, s_criterion, batch_size))

        train_dices.append(calculate_dice(inference_model, dataset_s_train))
        dev_dices.append(calculate_dice(inference_model, dataset_s_dev))
        eval_dices.append(calculate_dice(inference_model, dataset_eval_dev))

        segmentator.train()

    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model", "weights", "segmentator"+str(date_time)+".pth")
    torch.save(segmentator.state_dict(), model_path)

    util.plot_data([(s_train_losses, 'train_losses'), (s_dev_losses, 'dev_losses'), (d_train_losses, 'discriminator_losses'),
               (eval_domain_losses, 'eval_domain_losses')],
              'losses.png')
    util.plot_dice([(train_dices, 'train_dice'), (dev_dices, 'dev_dice'), (eval_dices, 'eval_dice')],
              'dices.png')

    inference_model = nn.Sequential(segmentator, selector, sigmoid)
    inference_model.to(device)
    inference_model.eval()

    print('Dice on annotated: ', calculate_dice(inference_model, dataset_s_test))
    print('Dice on unannotated: ', calculate_dice(inference_model, dataset_eval_test))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(234)
    torch.manual_seed(234)
    random.seed(234)
    torch.cuda.empty_cache()
    gc.collect()

    train_sources = [etlstream.Origin.SB]
    eval_source = etlstream.Origin.MC7
    train(train_sources, eval_source)