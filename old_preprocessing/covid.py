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
    x_ray = pickle.load(open('pre-processed-dataset/x_ray-pre-processed.pickle', 'rb'))


    print("Found data")

except IOError :
    print("Couldn't find data sets")
    # Read COVID data set
    covid_path = "covid-chestxray-dataset-master"
    image_covid_folder = os.path.join(covid_path, "images")
    image_list = os.listdir(image_covid_folder)
    # Remove CT images and not matching images between databases
    df_covid = pd.read_csv('covid-chestxray-dataset-master/metadata.csv')
    matching_df = df_covid[df_covid['filename'].isin(image_list)]
    xray_list = matching_df[matching_df['modality'] == 'X-ray']['filename'].tolist()
    # Assign labels of covid
    print('Xray list length', len(xray_list))
    xray_labels = df_covid[df_covid['filename'].isin(xray_list)]['finding'].values
    images_PIL = [Image.open(os.path.join(image_covid_folder, image_name)) for image_name in xray_list]
    covid = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True, covid = True)
    storedPickle = myData(covid,xray_labels)
    pickle.dump(storedPickle, open('pre-processed-dataset/covid-pre-processed.pickle', 'wb'))
    print('Saved covid pickles')
    covid_data = storedPickle.data
    covid_labels= storedPickle.labels

    train,train_l,val,val_l,test,test_l = splitData(covid_data,covid_labels)
    storedPickle_train = myData(train,train_l)
    storedPickle_val = myData(val,val_l)
    storedPickle_test = myData(test,test_l)
    pickle.dump(storedPickle_train, open('pre-processed-dataset/covid-pre-processed-train.pickle', 'wb'))
    pickle.dump(storedPickle_val, open('pre-processed-dataset/covid-pre-processed-val.pickle', 'wb'))
    pickle.dump(storedPickle_test, open('pre-processed-dataset/covid-pre-processed-test.pickle', 'wb'))
    print('Saved covid pickles')

print('Cleaned Data')

def read_txt(type='train'):
    if type=='train':
        np.random.seed(42)
        file = open('train_val_list.txt','r')
        data = file.read()
        list_of_names = data.split("\n")
        size = len(list_of_names)
        train_size = int(0.8*size)
        indeces = np.random.choice(np.arange(size),train_size,replace=False)
        indeces_val = list(set(np.arange(size)) - set(indeces))
        train = [list_of_names[index] for index in list(indeces)]
        val = [list_of_names[index] for index in list(indeces_val)]
        return train,val


    else:
        file = open('test_list.txt', 'r')
        data = file.read()
        list_of_names = data.split("\n")

        return list_of_names

    
train,val = read_txt()

test = read_txt('test')
# def getLabels(list):
#     df = pd.read_csv('Data_Entry_2017.csv')
#     label_list = []
#     counter=0
#     print(len(list))
#     for image_name in list:
#         if counter%100==0:
#             print(counter)
#         label_list.append(df[df['Image Index'] == image_name]['Finding Labels'].values[0].split('|')[0])
#         counter = counter+1
#     return label_list
#
# labels= getLabels(test)

f=open('val.txt','w')
for ele in val:
    f.write(ele+'\n')

f.close()
# print(len(labels))
# print(labels[0])



