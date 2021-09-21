import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(),
                                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU()
                                    )

    def forward(self, x):
        return self.layers(x)
