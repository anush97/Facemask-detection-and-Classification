# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:11:22 2022

@author: S_ANGH
"""


import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


# loading dataset
#data = pd.read_csv('G:/AI Project/mask-classification-pytorch/dataset.csv')
#data.head()

data_path = "G:/AI Project/mask-classification-pytorch/dataset.csv"
img_path = "G:\\AI Project\\mask-classification-pytorch\\dataset\\"


from MaskDataset import MaskImageDataset

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

full_dataset = MaskImageDataset(data_path, img_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])




train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)

#train_features, train_labels = next(iter(train_data))

# Data preprocessing ended
#######################################################


################# Model 2 ##############################


import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from CustomModel import MaskNetV2

model2 = MaskNetV2()
print(model2)

model2 = model2.to(device)

print(device)

##########################################

def trainModel2(model, train_loader, device):
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    
    train_losses = []
    acc_list = []
    epochs = 44
    
    for i in range(1, epochs+1):
        start = time.time()
        
        running_loss = 0.0
        total = 0
        correct = 0
        
        for j, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1) 
            correct += (predicted == labels).sum().item() 
        
            
        
        train_loss = running_loss/len(train_loader.sampler)
        train_losses.append(train_loss)
        accuracy = (correct / total) * 100
        acc_list.append(accuracy)
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tAccuracy: {:.6f}'.format(
        i, train_loss, accuracy))
        elapsed = time.time() - start
        print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:11], time.gmtime(elapsed)))
    
    print("Finished Training")
    torch.save(model.state_dict(), "G:/AI Project/mask-classification-pytorch/saved_models/masknetv2_3_dict.pt")
    torch.save(model, "G:/AI Project/mask-classification-pytorch/saved_models/masknetv2_3_full.pt")

#torch.save(model2.state_dict(), "G:/AI Project/mask-classification-pytorch/saved_models/m2.pt")

trainModel2(model2, train_data, device)