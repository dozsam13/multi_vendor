import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32), num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], num_class, 1), nn.Sigmoid())

    def forward(self, x):
        enc_ftrs = self.encoder(x)[::-1]
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        out = self.head(out)

        return out