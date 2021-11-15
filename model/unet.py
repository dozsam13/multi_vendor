import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        #self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], num_class, 1), nn.Sigmoid())
        self.head = nn.Conv2d(dec_chs[-1], 1, 1)

    def forward(self, x):
        encoder_activations = self.encoder(x)[::-1]
        out = self.decoder(encoder_activations[0], encoder_activations[1:])
        out = self.head(out)

        return out, encoder_activations[0]
