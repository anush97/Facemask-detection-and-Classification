# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:56:25 2022

"""

import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np

from sklearn.model_selection import KFold
from tqdm import tqdm

data_path = "G:/AI Project/mask-classification-pytorch/dataset.csv"
img_path = "G:/AI Project/mask-classification-pytorch/dataset/"

from MaskDataset import MaskImageDataset


transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()
                                ])

full_dataset = MaskImageDataset(data_path, img_path, transform=transform)

fDataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
# Try - 1
'''images, labels = next(iter(fDataloader))

mean, std = images.mean([0,2,3]), images.std([0,2,3])

print(mean)

print("-------")

print(std)'''

# Try 2 - Sejal's code 

mean = 0.0
meansq = 0.0
count = 0

for data, _ in tqdm(fDataloader):
    mean += data.sum(axis=[0,2,3])
    meansq += (data**2).sum(axis=[0,2,3])
    count += np.prod(data.shape)

total_mean = mean/count
total_var = (meansq/count) - (total_mean**2)
total_std = torch.sqrt(total_var)
print("mean: " + str(total_mean))
print("std: " + str(total_std))
