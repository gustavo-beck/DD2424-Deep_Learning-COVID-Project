# Import packages and methods
import preprocessing as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
from collections import Counter


class myData:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels

def read_images_names(path, path_labels):
    file = open(path,'r')
    data = file.read()
    images_names = data.split("\n")
    file_labels = open(path_labels,'r')
    data_labels = file_labels.read()
    images_labels = data_labels.split("\n")
    return images_names, len(images_names), images_labels

def preprocess_datasets(set_size, batch_size, set_images, set_labels, x_ray_path, dataset_name):
    # Define sizes
    split_size = int(set_size/batch_size)
    print(set_size)
    left_set = set_size - split_size * batch_size
    print(left_set)
    for i in range(split_size):
        start = i * batch_size 
        end = start + batch_size
        image_x_ray_list = set_images[start:end]
        xray_labels = set_labels[start:end]
        images_PIL = [Image.open(os.path.join(x_ray_path, image_name)) for image_name in image_x_ray_list]
        x_ray = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True, xray = True)
        xray_data = myData(x_ray,xray_labels)
        pickle.dump(xray_data , open('pre-processed-dataset/x_ray-pre-processed' + str(i) + '-' + dataset_name + '.pickle', 'wb'))
        print('Saved x-rays pickles', i)
    start = left_set
    image_x_ray_list = set_images[-start:]
    xray_labels = set_labels[-start:]
    images_PIL = [Image.open(os.path.join(x_ray_path, image_name)) for image_name in image_x_ray_list]
    x_ray = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True, xray = True)
    xray_data = myData(x_ray,xray_labels)
    pickle.dump(xray_data , open('pre-processed-dataset/x_ray-pre-processed-' + dataset_name + '.pickle', 'wb'))
    print('Saved x-rays pickles', i)

def splitData(images,labels,train_percentage=0.6):
    np.random.seed(42)
    dataset_size = len(images)
    training_size = int(dataset_size*train_percentage)
    validation_percentage = (1-train_percentage)/2
    validation_size = int(dataset_size*validation_percentage)
    test_size = dataset_size-training_size-validation_size
    total_indeces = np.arange(len(images))
    indeces_tr = np.random.choice(total_indeces,training_size,replace=False)
    remaining_indeces = list(set(total_indeces) - set(indeces_tr))
    indeces_val = np.random.choice(remaining_indeces,validation_size,replace=False)
    indeces_test = list(set(remaining_indeces)-set(indeces_val))


    train = [images[index] for index in list(indeces_tr)]
    train_l = [labels[index] for index in list(indeces_tr)]
    validation = [images[index] for index in list(indeces_val)]
    validation_l = [labels[index] for index in list(indeces_val)]
    test = [images[index] for index in list(indeces_test)]
    test_l = [labels[index] for index in list(indeces_test)]

    print('Train size', len(train))
    print('validation size', len(validation))
    print('Test size', len(test))
    print('All data',dataset_size)
    print('are they equal?', (len(train) + len(validation) + len(test)==dataset_size))
    return train,train_l,validation,validation_l,test,test_l

try:
    covid = pickle.load(open('pre-processed-dataset/covid-pre-processed.pickle', 'rb'))
    print("Found data")

except IOError :
    print("Couldn't find data sets")
    batch_size = 5000 
    # Read X-Ray data set
    x_ray_path = "all_100k_images/images"
    train_images, train_size, train_labels = read_images_names('train.txt', 'train_labels.txt')
    test_images, test_size, test_labels = read_images_names('test.txt', 'test_labels.txt')
    val_images, val_size, val_labels = read_images_names('val.txt', 'val_labels.txt')

    preprocess_datasets(val_size, batch_size, val_images, val_labels, x_ray_path, 'val')
    preprocess_datasets(train_size, batch_size, train_images, train_labels, x_ray_path, 'train')
    preprocess_datasets(test_size, batch_size, test_images, test_labels, x_ray_path, 'test')

print('Cleaned Data')


