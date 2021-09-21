import torch.nn as nn
import torch
from model.block import Block


class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64, 32)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.upconv_batchnorms = nn.ModuleList([nn.BatchNorm2d(chs[i]) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)

            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.upconv_batchnorms[i](x)
            x = self.dec_blocks[i](x)
        return x