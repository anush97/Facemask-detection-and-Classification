# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:45:54 2022

@author: S_ANGH
"""

import torch

from torchvision import transforms
from PIL import Image
from CustomModel import MaskNetV2
import matplotlib.pyplot as plt


def predictImage(model, imagePath, device, labels={0:'cloth', 1:'N95', 2:'N95 with valve', 3:'No Mask', 4:'Surgical'}):
    
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    image = Image.open(imagePath).convert('RGB')
    imageD = Image.open(imagePath).convert('RGB')
    
    image = transform(image)
    
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))
    pred = output.argmax(dim=1).cpu().numpy()
    #print(pred)
    
    plt.imshow(imageD)    
    print("Prediction : " + labels[pred[0]])
    



model = MaskNetV2()
model = torch.load("saved_models\masknetv2_3_full.pt")
img_path = "try.jpg"

predictImage(model, img_path, "cuda")