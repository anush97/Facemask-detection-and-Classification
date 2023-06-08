import torch
from torch.nn.functional import cross_entropy
import itertools
import numpy as np
import matplotlib.pyplot as plt

def test_model(model,testing_data,DEVICE):
  
    testing_loss = 0
    correct_prediction = 0 
    data_size = 0

    for batch in testing_data:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)          
            data_size += len(images)


            prediction = model(images)

            testing_loss += cross_entropy(prediction, labels).item()
            correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()


    accuracy = correct_prediction/data_size
    testing_loss = testing_loss/data_size

    print('\nTesting:')
    print(f"Correct prediction: {correct_prediction}/{data_size} and accuracy: {accuracy} and loss: {testing_loss}")


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




