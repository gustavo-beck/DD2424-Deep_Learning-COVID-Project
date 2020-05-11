import numpy as np
import pickle
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd

class myData:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels

def distance(grid, sample):
    return np.sum((grid - sample)**2, axis=2)

class SOM:
    def __init__(self, inputs, grid_width, grid_height, max_learning_rate = 0.8, min_learning_rate = 0.05, radious = 10): # input should be flattened
        np.random.seed(42)
        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        self.grid_height = grid_height
        self.grid_width = grid_width
        #self.grid = np.random.randn(grid_dim, grid_dim, mean.shape[0])*std*0.5 + mean
        # Try different initialization
        self.grid = np.random.normal(mean, std*0.25, (grid_width, grid_height, mean.shape[0]))
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.radious = radious
        
    def find_winner(self, samples):
        winners = np.zeros((len(samples), 2))
        for i, sample in enumerate(samples):
            dis = distance(self.grid, sample)
            winners[i] = np.unravel_index(np.argmin(dis), dis.shape)
        return winners

    def update_grid(self, samples, lr, current_radious):
        winners = self.find_winner(samples)
        for (win_x, win_y), sample in zip(winners, samples):
            for candidate_x, candidate_y in np.ndindex(self.grid_width, self.grid_height):
                distance_from_winner = (candidate_x - win_x)**2 + (candidate_y - win_y)**2
                if distance_from_winner <= current_radious**2:
                    self.grid[candidate_x, candidate_y] += lr*(sample - self.grid[candidate_x, candidate_y])*np.exp(-distance_from_winner / 2*current_radious**2)
            
    def train(self, inputs, epochs, batch_size):
        tau = epochs**2/(np.log(4*self.radious))
        for epoch in tqdm(range(epochs)): 
            current_radious = int(self.radious*np.exp(-(epoch)**2/tau))
            lr = self.max_learning_rate - (self.max_learning_rate - self.min_learning_rate)*epoch/epochs
            for i in tqdm(range(0, inputs.shape[0], batch_size)):
                batch = inputs[i:i+batch_size]
                self.update_grid(batch, lr, current_radious)
                

dictionary = {0:'Atelectasis', 1:'Cardiomegaly', 2:'Consolidation', 3:'Edema', 4:'Effusion', 5:'Emphysema', 6:'Fibrosis', 7:'Hernia', 8:'Infiltration', 9:'Mass', 10:'COVID-19', 11:'Nodule', 12:'Pleural_Thickening', 13:'Pneumonia', 14:'Pneumothorax'}      
path_for_soms = r'C:\Users\teodo\OneDrive\Desktop\KTH\4th_Period\Deep_Learning\Project\DD2424-Deep_Learning-COVID-Project\pytorch_example\balanced_data.pickle'
soms_dataset = pickle.load(open(path_for_soms, 'rb'))
indexes = np.random.permutation(soms_dataset.labels.shape[0])
soms_dataset.data = soms_dataset.data[indexes]
soms_dataset.labels = soms_dataset.labels[indexes]

df = pd.read_csv("xray_dataset/organized_dataset.csv")
df.pop('No Finding')  # Deleting the column no findings
cols = np.array(df.columns)
image_list = ['00000001_000.png',
        '00000040_001.png',
        '00000011_000.png',
        '00000111_000.png',
        '00000005_006.png', 
        '00000061_001.png',
        '00000008_002.png',
        '00000011_006.png',
        '00000071_001.png',
        '00000020_000.png',
        '00000061_015.png',
        '00000022_001.png',
        '00000032_024.png',
        '00030759_000.png']
model_path = 'final_model_2.pt'

def generateMap(model_path, pathImageFile, device):
    # Load model
    
    model= torch.load(model_path, map_location='cpu')
    model.to(device)
    model = model.features
    model.eval() # Doesn't store gradients

    # Prepare image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    imageData = Image.open(pathImageFile).convert('RGB')
    imageData = image_transform(imageData)
    imageData = imageData.unsqueeze_(0)
    imageData = imageData.to(device)

    # Walk the image through the model
    output = model(imageData)
    output = torch.nn.functional.relu(output)

    return output.reshape(-1).detach().numpy()
flatten = []
# Train SOM
grid_height = 10
grid_width = 10
epochs = 20
batch_size = 1
for i in range(len(image_list)):
    pathInputImage = 'images_small/' + image_list[i]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flatten.append(generateMap(model_path, pathInputImage, device))

flattened = np.array(flatten)
som = SOM(flattened, grid_width, grid_height)
som.train(flattened, epochs, batch_size)

# Plot the clusters, This is going to take much time
ranking = np.zeros((grid_width, grid_height, 15))
for sample, label in tqdm(zip(flattened, soms_dataset.labels)):
    pos_x, pos_y = som.find_winner([sample])[0]
    ranking[int(pos_x), int(pos_y), label] += 1
ranking_winners = np.array(np.argmax(ranking, axis=2), dtype=int)
number_of_points = np.sum(ranking, axis=2)
percentage_of_winner = np.max(ranking, axis=2)/number_of_points
number_of_points /= np.sum(number_of_points)


n = len(dictionary)
color= cm.rainbow(np.linspace(0,1,n))
c=np.array(color)
color_labels = np.zeros(len(dictionary))
fig, ax = plt.subplots()
for i, j in np.ndindex(grid_width, grid_height):
    label = dictionary[ranking_winners[i,j]]
    if color_labels[ranking_winners[i,j]] == 0: 
        plt.scatter(i,j, c=c[ranking_winners[i,j]], alpha=0.7, s=number_of_points[i,j]*1e4, edgecolors='none', label=label)
        color_labels[ranking_winners[i,j]] += 1
    else:
        plt.scatter(i,j, c=c[ranking_winners[i,j]], alpha=0.7, s=number_of_points[i,j]*1e4, edgecolors='none')


lgnd = plt.legend(markerscale=1, loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fontsize=10)
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])

# show the figure
fig.tight_layout()
plt.show()



