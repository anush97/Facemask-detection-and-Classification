import os
import torch, torchvision
import src.preprocess as preprocess
import src.CNN as CNN
import src.test as test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Loading Model and datasets 
# Copy and paste the model from workspace directory
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
MODEL_FILEPATH = os.path.join(os.path.abspath(os.curdir),'Model/model.pth')

DATASET_MALE = os.path.join(os.path.abspath(os.curdir),'Dataset/Dataset - Gender - Postbias/Test/Male')
DATASET_FEMALE = os.path.join(os.path.abspath(os.curdir),'Dataset/Dataset - Gender - Postbias/Test/Female')

DATASET_KIDS = os.path.join(os.path.abspath(os.curdir),'Dataset/Dataset - Age - Postbias/Test/0-18')
DATASET_MEN = os.path.join(os.path.abspath(os.curdir),'Dataset/Dataset - Age - Postbias/Test/18-55')
DATASET_ELDERS = os.path.join(os.path.abspath(os.curdir),'Dataset/Dataset - Age - Postbias/Test/55-100')


def loadDataset(DATASET_DIR):
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize([128,128]), #resizing every image into the desired values
            torchvision.transforms.RandomHorizontalFlip(), #Flips images horizontally with a probability of 0.5
            torchvision.transforms.ToTensor()   #size normalization and conversation to tensor
        ])     

    # Loads the images and labels from the specified folder and applies the given transformation
    data = torchvision.datasets.ImageFolder(DATASET_DIR, transform=transforms)                                       

    return torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

def evaluate(model, DATA):

    test.test_model(model,DATA,DEVICE)

    with torch.no_grad():
        labels_N_prediction = test.get_labels_N_prediction(model, DATA, DEVICE)

    print(classification_report(labels_N_prediction[0], labels_N_prediction[1], target_names = preprocess.CLASSES))
    conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
    plt.figure(figsize=(10, 10))
    test.display_confusion_matrix(conf_matrix, preprocess.CLASSES)


model = CNN.load_model(MODEL_FILEPATH,DEVICE)

MALE = loadDataset(DATASET_MALE)
FEMALE = loadDataset(DATASET_FEMALE)
TEEN = loadDataset(DATASET_KIDS)
MEN = loadDataset(DATASET_MEN)
OLD = loadDataset(DATASET_ELDERS)

print('MALE')
evaluate(model, MALE)
print('---------------------------')

print('FEMALE')
evaluate(model, FEMALE)
print('---------------------------')


print('TEEN')
evaluate(model, TEEN)
print('---------------------------')

print('MEN')
evaluate(model, MEN)
print('---------------------------')

print('OLD')
evaluate(model, OLD)
print('---------------------------')