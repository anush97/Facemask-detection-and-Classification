# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:27:45 2022

@author: S_ANGH
"""

import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from sklearn.model_selection import KFold


data_path = "G:/AI Project/mask-classification-pytorch/dataset.csv"
img_path = "G:/AI Project/mask-classification-pytorch/dataset/"

from MaskDataset import MaskImageDataset


transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.1880, 0.1771, 0.1691], std=[0.3164, 0.3016, 0.2958])
                                ])

full_dataset = MaskImageDataset(data_path, img_path, transform=transform)

'''train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])




train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)'''

#train_features, train_labels = next(iter(train_data))




################# Model ##############################


import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.nn.functional import cross_entropy

from CustomModel import MaskNetV2


##########################################

def trainModelWithKfold(model, train_loader, device, fold, test_loader):
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    
    train_losses = []
    acc_list = []
    epochs = 30
    
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
    
    print(f"Finished Training of fold {fold}")
    #torch.save(model.state_dict(), f"G:/AI Project/mask-classification-pytorch/saved_models/masknetv2_k_fold{fold}_dict.pt")
    torch.save(model, f"G:/AI Project/mask-classification-pytorch/saved_models/normal2_masknetv2_k_fold{fold}_full.pt")
    
    ########### Testing ######################
    testing_loss = 0
    correct_prediction = 0 
    data_size = 0
    prediction1=[]
    with torch.no_grad():
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)          
            data_size += len(images)
            prediction = model(images)
            
            prediction1.append(prediction)
            
            testing_loss += cross_entropy(prediction, labels).item()
            correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()


    test_accuracy = correct_prediction/data_size
    testing_loss = testing_loss/data_size

    print('\nTesting:')
    print(f"Correct prediction: {correct_prediction}/{data_size} and accuracy: {test_accuracy} and loss: {testing_loss}")
    
    return (accuracy, test_accuracy)


#torch.save(model2.state_dict(), "G:/AI Project/mask-classification-pytorch/saved_models/m2.pt")


def kFoldTraining(numFold, full_dataset):
    kfold = KFold(n_splits=numFold, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracies = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        
        print(f"Fold {fold}")
        
        trainSubSampler = SubsetRandomSampler(train_ids)
        testSubSampler = SubsetRandomSampler(test_ids)
        
        trainloader = DataLoader(full_dataset, batch_size=32, sampler=trainSubSampler)
        testloader = DataLoader(full_dataset, batch_size=32, sampler=testSubSampler)
        
        model = MaskNetV2()
        model.apply(resetWeight)
        model = model.to(device)
        
        
        train_accuracy, test_accuracy = trainModelWithKfold(model, trainloader, device, fold, testloader)
        
        accuracies.append((train_accuracy, test_accuracy))
        
    
    return accuracies


def resetWeight(model):
    
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
k_fold_accuracies = kFoldTraining(5, full_dataset)




