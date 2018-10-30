"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg
"""
import torch
import torch.nn as nn
from .wage_quantizer import WAGEQuantizer, Q
from .wage_initializer import wage_init_
import math

__all__ = ['VGG7LP']

class VGG(nn.Module):
    def __init__(self, wl_activate=-1, fl_activate=-1, wl_error=-1, fl_error=-1,
                 num_classes=10, depth=16, batch_norm=False, wl_weight=-1, writer=None):
        super(VGG, self).__init__()
        quant = lambda name : WAGEQuantizer(wl_activate, wl_error, name, writer=writer)
        self.features = nn.Sequential(*[
            # Turns out that the input quantization is never used in the original repo
            # Image input should already been quantized to 8-bits - no need do it again
            # WAGEQuantizer(wl_activate, -1, "input"), # only quantizing forward

            # Group 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            quant("feature-1-1"),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            quant("feature-1-2"),

            # Group 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            quant("feature-2-1"),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            quant("feature-2-2"),

            # Group 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            quant("feature-3-1"),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            quant("feature-3-2")
        ])

        self.classifier = nn.Sequential(
            nn.Linear(8192, 1024, bias=False),
            nn.ReLU(inplace=True),
            quant("classifier-lin"),
            nn.Linear(1024, num_classes, bias=False),
            WAGEQuantizer(-1, wl_error, "bf-loss") # only quantizing backward pass
        )

        self.weight_scale = {}
        self.weight_acc = {}
        for name, param in self.named_parameters():
            assert 'weight' in name
            wage_init_(param, wl_weight, name, self.weight_scale, factor=1.0)
            self.weight_acc[name] = Q(param.data, wl_weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Base:
    base = VGG
    args = list()
    kwargs = dict()

class VGG7LP(Base):
    kwargs = {'depth':7}

