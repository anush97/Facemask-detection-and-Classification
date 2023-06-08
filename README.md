# Mask Classification in Pytorch

## Required Libraries
- PyTorch (1.11) with Cuda
- Pandas
- NumPy
- Sci-kit Learn
- Matplotlib
- PIL
# -----------------------------------------------------------------------
Full Dataset: https://drive.google.com/drive/folders/1rLNzbcunez5B3hYQAiqorPD6Ov_lrLZT?usp=sharing

# -----------------------------------------------------------------------
## Files and their purpose:

1) CustomModel.py - This file contains the CNN model customized according to the requirements of the problem.
2) DataPreprocessTraining.py - This file contains all the functions that are required for loading the dataset along with training it by importing model from CustomModel.py and fetching dataset from MaskDataset.py. 
3) MaskDataset.py - Contains the code to read the file and image.
4) Evaluation.py- This file contains evaluation code ie testing the test dataset passing through thr trained model , along with calculating the accuracy , precision and recall etc.
5) Demo.py- Demo includes the code to classify the image we can randomly input.
6) Dataset / dataset.csv - This file contains the entire data set images and in the form of csv as well.
7) try1.jpg- trial image that can be passed in demo.py so as to find out which label classifies it.
8) AAI Project_Final_MaskNet.ipynb- It contains the entire code of all the modules from loading the dataset to evaluation , just in ipynb form.
9) saved_models- It is a folder containing our trained model.
# -----------------------------------------------------------------------
## WORKING OF THE PROJECT:

## Loading and Training

For Loading the dataset, all the functions are implemented in the DataPreproccessTraining.py. This file will load the dataset and train a CNN model on that dataset. There is MaskDataset.py for creating a custom dataset for pytorch framework. CustomModel.py is the custom CNN model architecture created in PyTorch. Before executing the below command, please change the paths for dataset and for saving the model 

```
python DataPreprocessTraining.py
```

This will save the trained model in saved_models folder.

## Evaluation

For evaluating the model, initially update the path of the trained model we saved after training along with that of the dataset of images in data_path and image_path respectively.Then run Evaluation.py file for various evaluations performed by passing an image via the trained model. Such evaluation contain stats like accuracy, precision, recall, support and confusion matrix. A detailed classification report is also generated.

```
python Evaluation.py
```

## Testing 

For testing the trained model, use Demo.py. Change the path of the testing image in the Demo.py file under the variable img_path. It is better to execute testing inside an IDE rather than executing it from commandline. So for different images just execute the following line in the IDE after loading the model and the function,

```Python
predictImage(model, img_path, "cuda")
```

## Alternate way to run this Project

Just execute the cells of the jupyter notebook with high RAM named AAI Project_Final_MaskNet.ipynb


## Phase - 2 K-Fold Training

For phase 2 of k-fold training run the Phase2_K_Fold.py to run k-fold training. Currently the value of k is set to 5 and each fold is trained for 30 epochs. On each fold the fold is saved in saved_models directory.

```
python Phase2_K_Fold.py
```

