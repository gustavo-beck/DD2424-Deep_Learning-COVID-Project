from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
import tensorflow as tf
import sys, os,cv2
from sklearn.utils import shuffle
import statistics
from statistics import mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class myData:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels


dictionary = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3, 'Effusion': 4, 'Emphysema': 5, 'Fibrosis': 6, 'Hernia': 7, 'Infiltration': 8, 'Mass': 9, 'COVID-19': 10, 'Nodule': 11, 'Pleural_Thickening': 12, 'Pneumonia': 13, 'Pneumothorax': 14}
path_for_soms = r'C:/Users/teodo/OneDrive/Desktop/KTH/4th_Period/Deep_Learning/Project/DD2424-Deep_Learning-COVID-Project/pytorch_example/balanced_data.pickle'
soms_dataset = pickle.load(open(path_for_soms, 'rb'))
indexes = np.random.permutation(soms_dataset.labels.shape[0])
soms_dataset.data = soms_dataset.data[indexes]
soms_dataset.labels = soms_dataset.labels[indexes]
print(np.unique(soms_dataset.labels, return_counts=True))
# flatten_dataset = soms_dataset.data.reshape(soms_dataset.data.shape[0], -1)

class SOM_Layer(): 

    def __init__(self,m,n,dim,learning_rate_som = 0.04,radius_factor = 1.1,gaussian_std=0.5):
        
        self.m = m
        self.n = n
        self.dim = dim
        self.gaussian_std = gaussian_std
        self.map = tf.Variable(tf.random.normal(shape=[m*n,dim],stddev=0.05)) # [900 x 1024]
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = learning_rate_som
        self.sigma = max(m,n)*1.1

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self):
        return self.map
    def getlocation(self):
        return self.bmu_locs

    def feedforward(self,input):
    
        self.input = input
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.input, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1) # winner
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self,iter,num_epoch):

        # Update the weigths 
        radius = tf.subtract(self.sigma,
                                tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                    tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        self.bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2)

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, self.gaussian_std)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, alpha)
        
        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
            tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op,axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.div(self.numerator, self.denominator)
        self.update = tf.assign(self.map, self.new_weights)

        return self.update

def reshapeImages(images):

    # setting dim of the resize
    height = 32
    width = 32
    dim = (width, height)

    # reshaping
    reshaped = [cv2.resize(np.array(image), dim, interpolation=cv2.INTER_LINEAR) for image in images]

    return np.array(reshaped)

def val2onehot(val_array, classes):
    labels = np.zeros((len(val_array), classes))
    for ind,lbl in enumerate(val_array):
        labels[ind,lbl] = 1
    return labels

labels_one_hot = val2onehot(soms_dataset.labels, 15)
resized_images = reshapeImages(soms_dataset.data)
train_batch = resized_images.reshape(resized_images.shape[0], -1)
map_width  = 10
map_height  = 10
map_dim = 32 * 32
num_epoch = 100
batch_size = 100
train_label = soms_dataset.labels
SOM_layer = SOM_Layer(map_width, map_height, map_dim, learning_rate_som=0.8, radius_factor=0.4, gaussian_std = 0.03)

# create the graph
x = tf.placeholder(shape=[None,map_dim],dtype=tf.float32)
current_iter = tf.placeholder(shape=[],dtype=tf.float32)

# graph
SOM_layer.feedforward(x)
map_update=SOM_layer.backprop(current_iter,num_epoch)

# session
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    # start the training
    for iter in range(num_epoch):
        for current_train_index in range(0,len(train_batch),batch_size):
            currren_train = train_batch[current_train_index:current_train_index+batch_size]
            sess_results = sess.run(map_update,feed_dict={x:currren_train,current_iter:iter})
            print('Current Iter: ',iter,' Current Train Index: ',current_train_index,' Current SUM of updated Values: ',sess_results.sum(),end='\r' )
        print('\n-----------------------')

    # after training is done get the closest vector
    n_samples = 50
    counter = np.zeros(15)
    samples = np.empty((0, train_batch.shape[1]))
    batch_labels = np.empty(0)
    for sample, label in zip(train_batch, train_label):
        if counter[label] < n_samples:
            counter[label] += 1
            samples = np.vstack((samples, sample))
            batch_labels = np.hstack((batch_labels, label))
            if np.sum(counter) == n_samples*counter.shape[0]:
                break

    locations = sess.run(SOM_layer.getlocation(),feed_dict={x:samples})
    x1 = locations[:,0]; y1 = locations[:,1]
    index = batch_labels
    index = list(map(str, index))

    map_centroid = {}
    for i, loc in enumerate(locations):
        if loc.tobytes() not in map_centroid:
            map_centroid[loc.tobytes()] = [batch_labels[i]]
        else:
            map_centroid[loc.tobytes()].append(batch_labels[i])
    
    # Check the majority
    most_freq = {}
    for key, value in map_centroid.items():
        most_freq[key] = max(set(value), key=value.count)
    
    ## Plots: 1) Train 2) Test+Train ###
    plt.figure(1, figsize=(12,6))
    plt.subplot(121)
    plt.scatter(x1,y1)
    # Just adding text
    for key, value in most_freq.items():
        numpy_key = np.frombuffer(key, dtype = locations.dtype)
        plt.text(numpy_key[0], numpy_key[1], value, ha='center', va='center', 
        bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.title('Train X-Ray Images')
    plt.show()

