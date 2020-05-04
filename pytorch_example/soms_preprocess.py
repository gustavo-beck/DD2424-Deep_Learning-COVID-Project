from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
import random
from torch import nn
import pickle
from tqdm import tqdm

class myData:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels


print('libraries alright')
path="/home/theodor/Git_Projects/DD2424-Deep_Learning-COVID-Project/pytorch_example/dataset/X-ray_pickles/"
img_name=[]
label=[]
labels = ['train', 'val', 'test']
directories = {}
for label in labels:
    directories[label] = []
for root, _, files in os.walk(path):
    files.sort()
    files = sorted(files, key=lambda fl: len(fl))
    for file in files:
        for label in labels:
            if label in file:
                directories[label].append(root+"/"+file)


pickle_file = open(directories['train'][0], 'rb')
current_pickle = pickle.load(pickle_file)
diseases = np.unique(current_pickle.labels)
idx = np.arange(len(diseases))
class_dict = dict(zip(diseases, idx))

# we define all the transformations we want to do with the images
# RandomAffine does a serries of transformations as explained here https://www.mathworks.com/discovery/affine-transformation.html
# with those transformations, the network will be able to handle distorted input
# torchvision.transforms.ToTensor() converts image to tensor input
images_size = 224 # 224
our_transforms = torchvision.transforms.ToTensor()
# not necessary to define it, but it helps organize the input
# class is a child of torch.utils.data.Dataset
# getitem enables when we do CustomDatasetFromImages[index]
class CustomDatasetFromImages(torch.utils.data.Dataset):
    def __init__(self, pickle_dir_array, labels_dict, transforms=None):
        self.labels_dict = labels_dict
        self.transforms = transforms
        self.pickle_dir_array = pickle_dir_array
        # initializing variables
        self.current_pickle = None
        self.current_pickle_id = -1
        # calculating length
        self.load_pickle_file(-1)
        self.data_len = len(self.current_pickle.labels)
        # loading the first file
        self.load_pickle_file(0)
        self.samples_per_pickle = len(self.current_pickle.labels)
        self.data_len += self.samples_per_pickle*(len(pickle_dir_array) - 1)

        
    def load_pickle_file(self, file_num, shuffle=False):
        pickle_file = open(self.pickle_dir_array[file_num], 'rb')
        self.current_pickle = pickle.load(pickle_file)
        self.current_pickle_id = file_num

    def index_to_pic_pos(self, index):
        return index // self.samples_per_pickle, index % self.samples_per_pickle

    def __getitem__(self, index):
        pic, pos = self.index_to_pic_pos(index)
        if pic != self.current_pickle_id:
            self.load_pickle_file(pic)
        img_array = self.current_pickle.data[pos]
        if self.transforms is not None:
            img_array = self.transforms(img_array)
        image_label = self.current_pickle.labels[pos]
        return (img_array, self.labels_dict[image_label])
    def __len__(self):
        return self.data_len

# shuffling the samples
train_set = CustomDatasetFromImages(directories['train'], class_dict, transforms=our_transforms) # train[0], train[1], .. , train[15]
val_set = CustomDatasetFromImages(directories['val'], class_dict, transforms=our_transforms)
test_set = CustomDatasetFromImages(directories['test'][0:1], class_dict, transforms=our_transforms)
# The loader is used to slpit the input and validation to batches, it returns arrays with the input in batches

path="/home/theodor/Git_Projects/DD2424-Deep_Learning-COVID-Project/pytorch_example/dataset/Covid_pickles/covid-pre-processed.pickle"
xray = pickle.load(open(path, 'rb'))
print(np.unique(xray.labels, return_counts=True))
print(class_dict)
xray.data = np.array(xray.data)
covid = xray.data[(np.array(xray.labels) == 'COVID-19')]
n_samples = covid.shape[0]
images = np.zeros((n_samples*(len(class_dict.keys())-1), 224, 224))
labels = []
pos=0
counter = np.zeros(len(class_dict.keys()))
for dataset in [train_set, val_set, test_set]:
    for image, label in dataset:
        if not (label == 10): #if not (no finding) 
            if counter[label] < n_samples:
                counter[label] += 1
                images[pos] = image
                labels.append(label)
                pos += 1
                if pos == images.shape[0]:
                    break
labels = np.array(labels)
images = np.vstack((covid, images))
labels = np.hstack((np.full(n_samples, 10), labels))
soms_dataset = myData(images, labels)
path_for_soms = "/home/theodor/Git_Projects/DD2424-Deep_Learning-COVID-Project/pytorch_example/dataset/soms/"
pickle.dump(soms_dataset, open(path_for_soms + 'balanched_data.pickle', 'wb'))
print(np.unique(labels, return_counts=True))
