#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:33:31 2023

@author: vaishnavijanakiraman
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, last_kernal = 3, stride_last = 2):
        super(ResidualBlock, self).__init__()
        self.dropout = 0.1
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, mid_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        
        #Depth wise convolution
        self.conv2 = nn.Sequential(
                        nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, stride = 1, padding = 1, groups = mid_channels),
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        
        #Dialated and strided convolution
        self.conv3 = nn.Sequential(
                        nn.Conv2d(mid_channels, out_channels, kernel_size = last_kernal, stride = stride_last, padding = 1, dilation=2),
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

        self.layer1 = ResidualBlock(3, 16, 24, stride_last = 1)
        self.layer2 = ResidualBlock(24, 32, 48)
        self.layer3 = ResidualBlock(48, 64, 96, stride_last = 1)
        self.layer4 = ResidualBlock(96, 96, 10, last_kernal = 1, stride_last = 1)

        self.gap = nn.AvgPool2d(kernel_size=10)
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
    
   
   