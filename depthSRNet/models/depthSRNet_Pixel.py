# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# from math import sqrt
import math
import numpy as np 
import torch.nn.init as init
from basic_module import *

class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv2d(inChannels, growthRate, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class SingleBlock(nn.Module):
    def __init__(self, inChannels,growthRate,nDenselayer):
        super(SingleBlock, self).__init__()
        self.block= self._make_dense(inChannels,growthRate, nDenselayer)
        
    def _make_dense(self,inChannels,growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                
    def forward(self, x):
        out = self.block(x)
        return out

#
class SRNet_model(nn.Module):
    def __init__(self, inChannels, growthRate, nDenselayer, nBlock):
        super(SRNet_model, self).__init__()
        
        self.conv1 = nn.Conv2d(1, growthRate, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(growthRate, growthRate, kernel_szie=3, padding=1, bias=True)

        inChannels = growthRate
        self.denseblock = self._make_block(inChannels, growthRate, nDenselayer, nBlock)
        inChannels += growthRate*nDenselayer*nBlock
        
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=128, kernel_size=1, padding=0, bias=True)

        self.global_feature_2x = global_2x_4x(1, 128)
        # self.global_feature_4x = global_2x_4x(1, 128)
        # self.global_feature_8x = global_8x(1, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_block(self, inChannels, growthRate, nDenselayer, nBlock):
        blocks = []
        for i in range(int(nBlock)):
            blocks.append(SingleBlock(inChannels,growthRate,nDenselayer))
            inChannels += growthRate*nDenselayer
        return nn.Sequential(* blocks)  
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.denseblock(out)                                      
        out = self.Bottleneck(out)

        global_feature_2x = self.global_feature_2x(out)
        concat = torch.cat([out, global_feature_2x], 1)

        # global_feature_4x = self.global_feature_4x(out)
        # concat = torch.cat([out, global_feature_4x], 1)
        #
        # global_feature_8x = self.global_feature_8x(out)
        # concat = torch.cat([out, global_feature_8x], 1)

        HR = self.up_xx(concat)
        return HR
