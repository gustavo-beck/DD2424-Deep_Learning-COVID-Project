import numpy as np
import pickle
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.pyplot import cm
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import itertools
from collections import OrderedDict
from functools import partial
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

class myData:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels

# Reading Covid
path2covid = 'covid_dataset/'
df_covid = pd.read_csv(path2covid + "metadata.csv")
df_covid = df_covid[(df_covid['modality'] == "X-ray") & (df_covid['view'] != "L") & (df_covid['finding'] == "COVID-19")]
df_covid = df_covid[['finding', 'filename']]
covid_list = df_covid['finding'].values
covid_image_list = df_covid['filename'].values
amount_of_covid = len(df_covid)
path2covid_images = 'covid_dataset/images/'

# Read the other diseases dataset
df = pd.read_csv("xray_dataset/organized_dataset.csv")
df.pop('No Finding')  # Deleting the column no findings
cols = np.array(df.columns)
diseases_list = cols[2:]
df.loc[:,'sum'] = df[cols[2:]].sum(axis=1)
df = df[df['sum'] == 1]
df.pop('sum')
df.pop('Dataset')

image_list = []
image_labels = []
sample_num = amount_of_covid
counter = np.zeros(len(diseases_list))
for i, row in tqdm(df.iterrows()):
    disease_index = np.argmax(list(row[1:].values))
    disease_name = diseases_list[disease_index]
    if counter[disease_index] < sample_num:
        counter[disease_index] += 1
        image_list.append(row[0])
        image_labels.append(disease_name)

image_list.extend(covid_image_list)
image_labels.extend(covid_list)

try:
    print("Findind data")
    data = pickle.load(open('data_som.pickle', 'rb'))
except IOError:
    print("Couldn't find data")
    model_path = 'final_model_16.pt'

    def generateMap(model, pathImageFile, device):

        # Prepare image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = image_transform(imageData)
        imageData = imageData.unsqueeze_(0)
        imageData = imageData.to(device)

        # Walk the image through the model
        output = model(imageData)
        #plt.imshow(output.detach().numpy().reshape((32,32)))
        #plt.show()
        flat_output = output.detach().numpy()
        return flat_output

    flatten = []
    print("START WALKING THE IMAGES THROUGH THE MODEL")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= torch.load(model_path, map_location='cpu')
    model.to(device)
    model.classifier = torch.nn.Identity()
    model.eval() # Doesn't store gradients
    for i in tqdm(range(len(image_list))):
        if image_labels[i] == 'COVID-19':
            pathInputImage = path2covid_images + image_list[i]
        else:
            pathInputImage = 'images_small/' + image_list[i]
        flatten.append(generateMap(model, pathInputImage, device))

    flattened = np.array(flatten)
    data = myData(flattened, image_labels)
    pickle.dump(data, open('data_som.pickle', 'wb'))

print("start Mani-folds")
# Shuffling the data
data_flattened = data.data # array
data_labels = data.labels # list
data_flattened = data_flattened.reshape(-1, data_flattened.shape[2])
all_diseases_labels = set(image_labels)
image_list = []
image_labels = []
sample_num = 10
counter = np.zeros(len(all_diseases_labels))
dictionary = {'Atelectasis':0, 'Cardiomegaly':1, 'Consolidation':2, 'Edema':3, 'Effusion':4, 'Emphysema':5, 'Fibrosis':6, 'Hernia':7, 'Infiltration':8, 'Mass':9, 'COVID-19':10, 'Nodule':11, 'Pleural_Thickening':12, 'Pneumonia':13, 'Pneumothorax':14}
for i in range(len(data_flattened)):
    disease_index = dictionary[data_labels[i]]
    if counter[disease_index] < sample_num:
        counter[disease_index] += 1
        image_list.append(data_flattened[i])
        image_labels.append(data_labels[i])


data_flattened = np.array(image_list)
data_labels = image_labels
n_neighbors = 10
n_components = 2

# Plot the clusters
print("START CLUSTERS PLOTS")
n_points = len(data_flattened)
n = len(all_diseases_labels)
color_range = cm.rainbow(np.linspace(0,1,n))
color_range = np.array(color_range)
color = []
for i in range(len(data_labels)):
    color.append(color_range[dictionary[data_labels[i]]])
# Set-up manifold methods
methods = OrderedDict()
#LLE = partial(manifold.LocallyLinearEmbedding,
              #n_neighbors, n_components, eigen_solver='auto')
#methods['LLE'] = LLE(method='standard')
#methods['LTSA'] = LLE(method='ltsa')
#methods['Hessian LLE'] = LLE(method='hessian')
#methods['Modified LLE'] = LLE(method='modified')
#methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
#methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
#methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           #n_neighbors=n_neighbors)
methods['t-SNE'] = manifold.TSNE(n_components=n_components,
                                perplexity=25,
                                learning_rate= 777.0,
                                n_iter=1000000,
                                n_iter_without_progress = 1000,
                                min_grad_norm=1e-07,
                                metric='euclidean',
                                init='pca',
                                verbose=0,
                                random_state=42,
                                method='exact',
                                angle=0.5,
                                n_jobs=-1)

# Plot results
color_labels = np.zeros(len(all_diseases_labels))
marker = itertools.cycle((',', '+', '.', 'o', '*'))
disease_marker = []
for i in range(n):
    disease_marker.append(next(marker))

for i, (label, method) in tqdm(enumerate(methods.items())):
    t0 = time()
    Y = method.fit_transform(data_flattened)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    sample_num = 1
    counter = np.zeros(len(all_diseases_labels))
    for j in range(len(data_labels)):
        disease_index = dictionary[data_labels[j]]
        if counter[disease_index] == 0:
            plt.scatter(Y[j, 0], Y[j, 1], c=color[j], marker = disease_marker[disease_index], cmap=plt.cm.Spectral, label = data_labels[j])
            counter[disease_index] += 1
        else:
            plt.scatter(Y[j, 0], Y[j, 1], c=color[j], marker = disease_marker[disease_index], cmap=plt.cm.Spectral)

plt.legend(markerscale=1, loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fontsize=10)
plt.tight_layout()
plt.show()