import torch
import torch.nn as nn
from model.encoder import Encoder


class Discriminator(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128, 256, 512), n_domains=2):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.head = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  View(32768),
                                  nn.Linear(32768, n_domains)
                                  )

    def forward(self, x, enc_repr):
        enc_ftr = self.encoder(x)[::-1][0]  #512*16*16
        enc = torch.cat([enc_repr, enc_ftr], dim=1)
        dom = self.head(enc)

        return dom


class View(nn.Module):
    def __init(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(-1, self.shape)
