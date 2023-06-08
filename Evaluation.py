import torch
from torch.utils.data import  DataLoader,random_split
from torchvision import datasets,transforms
import torch.nn as nn

from CustomModel import MaskNetV2
from MaskDataset import MaskImageDataset
import numpy as np

import matplotlib.pyplot as plt


data_path = "G:/AI Project/mask-classification-pytorch/dataset.csv"
img_path = "G:\\AI Project\\mask-classification-pytorch\\dataset\\"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
#Loading Model 

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
the_model = MaskNetV2()
path = "G:/AI Project/mask-classification-pytorch/masknetv2_3_full.pt"
the_model=torch.load(path)
the_model = the_model.to(device)
#the_model=torch.load("G:/AI Project/mask-classification-pytorch/masknetv2_3_full.pt",torch.device('cpu'))

from torch.nn.functional import cross_entropy
def test_model(model,testing_data,DEVICE):
      
    testing_loss = 0
    correct_prediction = 0 
    data_size = 0
    prediction1=[]
    for images, labels in testing_data:
            images = images.to(device)
            labels = labels.to(device)          
            data_size += len(images)
            prediction = model(images)
            
            prediction1.append(prediction)
            
            testing_loss += cross_entropy(prediction, labels).item()
            correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()


    accuracy = correct_prediction/data_size
    testing_loss = testing_loss/data_size

    print('\nTesting:')
    print(f"Correct prediction: {correct_prediction}/{data_size} and accuracy: {accuracy} and loss: {testing_loss}")


test_model(the_model,test_data,device)

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


# Evaluation
with torch.no_grad():
    labels_N_prediction = get_labels_N_prediction(the_model, test_data, device)

    
print(classification_report(labels_N_prediction[0], labels_N_prediction[1]))
conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
print(conf_matrix)

#%%
import pandas as pd
#ACCURACY SCORE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib
#y_pred = the_model.predict(test_dataset)
y_pred =labels_N_prediction[1]
y_test =labels_N_prediction[0]
print("Accuracy : ",accuracy_score(y_test, y_pred))

#PRECISION , RECALL,FSCORE,SUPPORT
from sklearn.metrics import precision_recall_fscore_support
d=()
d=precision_recall_fscore_support(labels_N_prediction[0], y_pred)
prec = d[0].tolist()
recall = d[1].tolist()
fscore = d[2].tolist()
support = d[3].tolist()
#precision_recall_fscore_support(y_test, y_pred,average="macro")
print("prec --",prec)
print("recall --",recall)
print("fscore --",fscore)
print("support --",support)

#CLASSIFICATION REPORT - PRECISION , RECALL,FSCORE,SUPPORT
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#PLOTTING CLASSIFICATION REPORT
import seaborn as sns
from pylab import savefig
h= classification_report(y_test, y_pred , output_dict=True)
svm =sns.heatmap(pd.DataFrame(h).iloc[:-1, :].T, annot=True)
figure = svm.get_figure()

figure.savefig('Classification report.png')

#CONFUSION MATRIX & PLOTTING IT
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix = confusion_matrix(y_test, y_pred).tolist()
cm = sns.heatmap(confusion_matrix, annot=True, fmt='d')


cm.plot()
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('confusion metrics.png')

#PLOTTING PRECISION AND RECALL GRAPH
from sklearn.metrics import PrecisionRecallDisplay

disp = PrecisionRecallDisplay(precision=precision_recall_fscore_support(y_test, y_pred)[0],recall=precision_recall_fscore_support(y_test, y_pred)[1])
disp.plot()
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('Precision vs Recall.png')