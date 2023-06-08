# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:18:34 2022

@author: S_ANGH
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MaskImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
            
        return image, label