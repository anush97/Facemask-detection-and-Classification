# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:28:21 2022

@author: S_ANGH
"""

from torch.nn import Conv2d, Linear, MaxPool2d, Module, BatchNorm2d
from torch.nn import functional as F



class MaskNetV2(Module):
    
    def __init__(self):
        ''' Initializing the model'''
        super(MaskNetV2, self).__init__()
        
        self.conv1_1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn_1 = BatchNorm2d(32)
        
        self.conv2_1 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn_2 = BatchNorm2d(64)
        
        self.conv3_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.bn_3 = BatchNorm2d(128)
        
        self.conv4_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.bn_4 = BatchNorm2d(256)
        
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = Linear(36864, 1024)
        self.fc2 = Linear(1024, 512)
        self.fc3 = Linear(512, 5)
        
        
    def forward(self, x):
        
        x = F.relu(self.bn_1(self.conv1_1(x)), inplace=True)
        x = self.maxpool(x)
        
        x = F.relu(self.bn_2(self.conv2_1(x)), inplace=True)
        x = self.maxpool(x)
        
        x = F.relu(self.bn_3(self.conv3_1(x)), inplace=True)
        x = self.maxpool(x)
        
        x = F.relu(self.bn_4(self.conv4_1(x)), inplace=True)
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.3)
        
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, 0.3)
        
        x = self.fc3(x)
        return x