#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 06:49:52 2023

@author: vaishnavijanakiraman
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernels = [3,3,1], strides = [1,1,1], paddings=[1,1,0]):
        super(ResidualBlock, self).__init__()
        self.dropout = 0.1
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, mid_channels, kernel_size = kernels[0], stride = strides[0], padding = paddings[0]),
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        
        #Depth wise convolution
        self.conv2 = nn.Sequential(
                        nn.Conv2d(mid_channels, mid_channels, kernel_size = kernels[1], stride = strides[1], padding = paddings[1], groups = mid_channels),
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        
        #Dialated and strided convolution
        self.conv3 = nn.Sequential(
                        nn.Conv2d(mid_channels, out_channels, kernel_size = kernels[2], stride = strides[2], padding = paddings[2], dilation=2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
    
    
    
class Net(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()

        self.layer1 = ResidualBlock(3, 48, 64,  kernels = [7,3,3], strides = [1,1,1], paddings = [0,0,0])
        self.layer2 = ResidualBlock(64, 128, 12, kernels = [3,3,1], strides = [1,1,1])
        self.layer3 = ResidualBlock(12, 48, 64, strides = [1,1,1])
        self.layer4 = ResidualBlock(64, 128, 10, kernels = [3,3,1], strides = [1,1,1], paddings = [0,0,0])

        self.gap = nn.AvgPool2d(kernel_size=16)
        #self.fc = nn.Linear(60, num_classes)
        
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x
    
   
   