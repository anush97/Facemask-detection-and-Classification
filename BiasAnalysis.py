# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:20:04 2022

@author: Anushka Sharma
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
from torch.utils.data import  DataLoader,random_split
from torchvision import datasets,transforms
import torch.nn as nn
import itertools
import sys
sys.path.append('/content/drive/MyDrive/mask-classfication')
from CustomModel import MaskNet , MaskNetV2
from MaskDataset import MaskImageDataset
import numpy as np

import matplotlib.pyplot as plt
import os
import torch, torchvision


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def loadDataset(DATASET_DIR,data_path):
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    full_dataset = MaskImageDataset(data_path, DATASET_DIR, transform=transform)                                   
   
    return DataLoader(full_dataset, batch_size=32, shuffle=True)

CLASSES = ['Cloth-Mask','N95-Mask','N95with Valve-Mask','No-Mask','Surgical-Mask']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

data_path_MALE = "/content/drive/MyDrive/mask-classfication/Male.csv"
data_path_FEMALE = "/content/drive/MyDrive/mask-classfication/Female.csv"
data_path_KIDS = "/content/drive/MyDrive/mask-classfication/Kids.csv"
data_path_MEN = "/content/drive/MyDrive/mask-classfication/Adults.csv"
data_path_ELDERS = "/content/drive/MyDrive/mask-classfication/Elder.csv"

DATASET_MALE = '/content/drive/MyDrive/mask-classfication/Dataset- kfold/Postbias/Gender/Male'
DATASET_FEMALE = '/content/drive/MyDrive/mask-classfication/Dataset- kfold/Postbias/Gender/Female'

DATASET_KIDS = '/content/drive/MyDrive/mask-classfication/Dataset- kfold/Postbias/Age/Kids'
DATASET_MEN = '/content/drive/MyDrive/mask-classfication/Dataset- kfold/Postbias/Age/Adults'
DATASET_ELDERS ='/content/drive/MyDrive/mask-classfication/Dataset- kfold/Postbias/Age/Elder'


from torch.nn.functional import cross_entropy
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot
def test_model(model,test_data,DEVICE):
    #print('RBYRY')  
    testing_loss = 0
    correct_prediction = 0 
    data_size = 0
    prediction1=[]
    #print('RBYRY1')
    #print(test_data)
        
    for images, labels in test_data:
            #print('RBYRY134')
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
    return test_accuracy
    #print(prediction1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def get_labels_N_prediction(model,loader,DEVICE):
    all_labels = []
    all_prediction = []

    for batch in loader:
        images, labels = batch
        images = images.to(DEVICE)

        prediction = model(images).to(torch.device("cpu")).argmax(dim=1).detach().numpy()
        labels = labels.to(torch.device("cpu")).detach().numpy()

        all_prediction = np.append(all_prediction,prediction)
        all_labels = np.append(all_labels,labels)

    return [all_labels,all_prediction]

def display_confusion_matrix(conf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(conf_matrix)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate2(model, DATA,device):
    #print('DVDV')
    t_acc = test_model(model,DATA,device)
    #print('MALE')
    with torch.no_grad():
        labels_N_prediction = get_labels_N_prediction(model, DATA, device)

    print(classification_report(labels_N_prediction[0], labels_N_prediction[1], target_names =CLASSES))
    conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
    plt.figure(figsize=(10, 10))
    display_confusion_matrix(conf_matrix,CLASSES)
    
    y_pred =labels_N_prediction[1]
    y_test =labels_N_prediction[0]
    #disp = PrecisionRecallDisplay(precision=precision_recall_fscore_support(y_test, y_pred)[0],recall=precision_recall_fscore_support(y_test, y_pred)[1])
    #isp.plot( marker='.')
    #matplotlib.pyplot.show()
    #matplotlib.pyplot.savefig('Precision vs Recall.png')

    prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)

    return (t_acc, prec, recall, fscore)


MALE = loadDataset(DATASET_MALE,data_path_MALE)
FEMALE = loadDataset(DATASET_FEMALE,data_path_FEMALE)
TEEN = loadDataset(DATASET_KIDS,data_path_KIDS)
MEN = loadDataset(DATASET_MEN,data_path_MEN)
OLD = loadDataset(DATASET_ELDERS,data_path_ELDERS)

bias_measures_male = []
bias_measures_female = []
bias_measures_teen = []
bias_measures_adult = []
bias_measures_old = []

for i in range(0, 10):

    model = MaskNetV2()
    model = model.to(device)
    path = f"/content/drive/MyDrive/mask-classfication/saved_models/masknetv2_k_fold{i}_full.pt"
    model = torch.load(path,torch.device('cpu'))

    print(f"For Model {i} \n")
    print('MALE')
    bias_measures_male.append(evaluate2(model, MALE,device))
    print('---------------------------')

    print('FEMALE')
    bias_measures_female.append(evaluate2(model, FEMALE,device))
    print('---------------------------')


    print('KIDS')
    bias_measures_teen.append(evaluate2(model, TEEN,device))
    print('---------------------------')

    print('ADULTS')
    bias_measures_adult.append(evaluate2(model, MEN,device))
    print('---------------------------')

    print('ELDER')
    bias_measures_old.append(evaluate2(model, OLD,device))
    print('---------------------------')
    

avg_acc_male = sum([x[0] for x in bias_measures_male]) / len(bias_measures_male)
avg_acc_female = sum([x[0] for x in bias_measures_female]) / len(bias_measures_female)
avg_acc_teen = sum([x[0] for x in bias_measures_teen]) / len(bias_measures_teen)
avg_acc_adult = sum([x[0] for x in bias_measures_adult]) / len(bias_measures_adult)
avg_acc_old = sum([x[0] for x in bias_measures_old]) / len(bias_measures_old)

avg_prec_male = sum([x[1] for x in bias_measures_male]) / len(bias_measures_male)
avg_prec_female = sum([x[1] for x in bias_measures_female]) / len(bias_measures_female)
avg_prec_teen = sum([x[1] for x in bias_measures_teen]) / len(bias_measures_teen)
avg_prec_adult = sum([x[1] for x in bias_measures_adult]) / len(bias_measures_adult)
avg_prec_old = sum([x[1] for x in bias_measures_old]) / len(bias_measures_old)

avg_recall_male = sum([x[2] for x in bias_measures_male]) / len(bias_measures_male)
avg_recall_female = sum([x[2] for x in bias_measures_female]) / len(bias_measures_female)
avg_recall_teen = sum([x[2] for x in bias_measures_teen]) / len(bias_measures_teen)
avg_recall_adult = sum([x[2] for x in bias_measures_adult]) / len(bias_measures_adult)
avg_recall_old = sum([x[2] for x in bias_measures_old]) / len(bias_measures_old)

avg_fsc_male = sum([x[3] for x in bias_measures_male]) / len(bias_measures_male)
avg_fsc_female = sum([x[3] for x in bias_measures_female]) / len(bias_measures_female)
avg_fsc_teen = sum([x[3] for x in bias_measures_teen]) / len(bias_measures_teen)
avg_fsc_adult = sum([x[3] for x in bias_measures_adult]) / len(bias_measures_adult)
avg_fsc_old = sum([x[3] for x in bias_measures_old]) / len(bias_measures_old)

print("Avg Accuracy Male :" + str(avg_acc_male))
print("Avg Accuracy Female: " + str(avg_acc_female))
print("Avg Accuracy Teen: " + str(avg_acc_teen))
print("Avg Accuracy Adult: " + str(avg_acc_adult))
print("Avg Accuracy Old: " + str(avg_acc_old))


print("Avg Precision Male: " + str(avg_prec_male))
print("Avg Precision female: " + str(avg_prec_female))
print("Avg Precision Teen: " + str(avg_prec_teen))
print("Avg Precision Adult: " + str(avg_prec_adult))
print("Avg Precision Old: " + str(avg_prec_old))

print("Avg Recall Male: " + str(avg_recall_male))
print("Avg Recall Female: " + str(avg_recall_female))
print("Avg Recall Teen: " + str(avg_recall_teen))
print("Avg Recall Adult: " + str(avg_recall_adult))
print("Avg Recall Old: " + str(avg_recall_old))

print("Avg Fscore Male: " + str(avg_fsc_male))
print("Avg Fscore Female: " + str(avg_fsc_female))
print("Avg Fscore Teen: " + str(avg_fsc_teen))
print("Avg Fscore Adult: " + str(avg_fsc_adult))
print("Avg Fscore Old: " + str(avg_fsc_old))


