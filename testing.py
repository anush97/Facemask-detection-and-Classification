#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

data_path = "C:/Users/ANUSHKA SHARMA/Desktop/Concordia/sem2/AAI/mask-classfication/dataset.csv"
img_path = "C:\\Users\\ANUSHKA SHARMA\\Desktop\\Concordia\\sem2\\AAI\\mask-classfication\\dataset\\"


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
# %%
#pip install skorch
from torch import nn
from skorch import NeuralNetClassifier


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms 

from skorch import NeuralNetClassifier

DEVICE = torch.device("cpu")

y_train = np.array([y for x, y in iter(train_dataset)])

class ConvNet(nn.Module):
    def __init__(self,dropout_rate=0.4,dropout_rate2=0.2,l1=50):
        # We optimize dropout rate in a convolutional neural network.
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop1=nn.Dropout2d(p=dropout_rate)  
        
        self.fc1 = nn.Linear(32 * 7 * 7, l1)
        self.drop2=nn.Dropout2d(p=dropout_rate2)
        
        self.fc2 = nn.Linear(l1, 10)

    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x),kernel_size = 2))
        
        x = F.relu(F.max_pool2d(self.conv2(x),kernel_size = 2))
        x = self.drop1(x)
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        
        x = self.drop2(x)
        x = self.fc2(x)
        return x
    
torch.manual_seed(0)

net = NeuralNetClassifier(
    ConvNet,
    max_epochs=10,
    iterator_train__num_workers=4,
    iterator_valid__num_workers=4,
    lr=1e-3,
    batch_size=64,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=DEVICE
)
net.fit(train_dataset, y=y_train)

val_loss=[]
train_loss=[]
for i in range(10):
    val_loss.append(net.history[i]['valid_loss'])
    train_loss.append(net.history[i]['train_loss'])
    
plt.figure(figsize=(10,8))
plt.semilogy(train_loss, label='Train loss')
plt.semilogy(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()    

# %%
from CustomModel import MaskNet

model = MaskNet()
model.load_state_dict(torch.load("G:/AI Project/mask-classification-pytorch/saved_model/m1.pt"))
model.eval()

# %%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

y_pred = net.predict(test_dataset)
y_test = np.array([y for x, y in iter(test_dataset)])
print(accuracy_score(y_test, y_pred))


from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(net,test_dataset, y_test.reshape(-1, 1))
plt.show()



from skorch.helper import SliceDataset

train_sliceable = SliceDataset(train_dataset)
scores = cross_val_score(net, train_sliceable,y_train, cv = 5, scoring = "accuracy")

print('validation accuracy for each fold: {}'.format(scores))
print('avg validation accuracy: {:.3f}'.format(scores.mean()))
#------------------

torch.manual_seed(0)
output = model(val_x.cuda())
softmax = torch.exp(output).cpu()
prob = list(softmax.detach().numpy())
predictions = np.argmax(prob, axis=1)
accuracy_score(val_y, predictions)