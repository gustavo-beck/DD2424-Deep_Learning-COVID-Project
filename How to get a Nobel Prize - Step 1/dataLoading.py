# This is the file in charge of loading data in MemeNet

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
from memeNet import *
import glob
from natsort import natsorted  # This library sorts a list in a "natural" way

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
        self.data_len += self.samples_per_pickle * (len(pickle_dir_array) - 1)

    def load_pickle_file(self, file_num, shuffle=False):
        pickle_file = open(self.pickle_dir_array[file_num], 'rb')
        self.current_pickle = pickle.load(pickle_file)
        pickle_file.close()  # Option: change to with open(...)
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