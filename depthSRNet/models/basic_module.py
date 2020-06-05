# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
import math

######################################################
def convnobn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False))

class BasicBlockNobn(nn.Module):
    def __init__(self, in_channels, out_channels, stride, pad, dilation):
        super(BasicBlockNobn, self).__init__()
        self.conv1 = nn.Sequential(convnobn(in_channels, out_channels, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
        self.conv2 = convnobn(out_channels, out_channels, 3, 1, pad, dilation)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        ori_x = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + ori_x
        out = self.relu2(out)
        return out

class global_2x_4x(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.resblock = BasicBlockNobn(out_channels, out_channels, 1, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_szie=3, padding=1, bias=True)
    def forward(self, x):
        out1 = self.conv1(x)
        outres = self.resblock(out1)
        out2 = self.conv2(outres)
        return out2

class global_8x(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.resblock_1 = BasicBlockNobn(out_channels, out_channels, 1, 1, 1)
        self.resblock_2 = BasicBlockNobn(out_channels, out_channels, 1, 1, 1)
        self.convt1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1,
                                         bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        outres1 = self.resblock_1(out1)
        outres2 = self.resblock_1(outres1)
        out2 = self.conv2(outres2)
        return out2

class up_xx(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.upxx = nn.Sequential(
            BasicBlockNobn(in_channels, out_channels, 1, 1, 1),
            BasicBlockNobn(out_channels, out_channels, 1, 1, 1),
            nn.PixelShuffle(2),  # (n, 64, 2*2*h, 2*2*w)
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
    def forward(self, x):
        out = self.upxx(x)
        return out



