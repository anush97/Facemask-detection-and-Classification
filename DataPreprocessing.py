
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



from PIL import Image

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

from skimage.transform import rotate
from skimage.util import random_noise


import torch

from torchvision import datasets, transforms

# loading dataset
data = pd.read_csv('G:/AI Project/mask-classification-pytorch/dataset.csv')
data.head()



# loading images
train_img = []
for img_name in tqdm(data['filename']):
    image_path = "G:\\AI Project\\mask-classification-pytorch\\dataset\\" + img_name
    # image_path='D:\\Concordia\\SEM2\\AI\\Project\\Dataset\\Dataset\\clothmask-22.png'
    img = Image.open(image_path)
    img = img.resize((256,256,3))
    img = np.array(img)
    img = img/255
    train_img.append(img)
    


train_x = np.array(train_img)
train_y = data['Category'].values
train_x.shape, train_y.shape


train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.25, random_state = 13, stratify=train_y)
(train_x.shape, train_y.shape), (test_x.shape, test_y.shape)

final_train_data = []
final_target_train = []
for i in tqdm(range(train_x.shape[0])):
    final_train_data.append(train_x[i])
    final_train_data.append(rotate(train_x[i], angle=45, mode = 'wrap'))
    final_train_data.append(np.fliplr(train_x[i]))
    # final_train_data.append(np.flipud(train_x[i]))
    final_train_data.append(random_noise(train_x[i],var=0.2**2))
    for j in range(5):
        final_target_train.append(train_y[i])
len(final_target_train), len(final_train_data)
final_train = np.asarray(final_train_data)
final_target_train = np.asarray(final_target_train)

'''fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
for i in range(5):
    ax[i].imshow(final_train[i+30])
    ax[i].axis('off')'''


final_train = final_train.reshape(7405, 3, 224, 224)
final_train  = torch.from_numpy(final_train)
final_train = final_train.float()

final_target_train = final_target_train.astype(int)
final_target_train = torch.from_numpy(final_target_train)




import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from CustomModel import MaskNet

model = MaskNet()