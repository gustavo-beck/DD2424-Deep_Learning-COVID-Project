import numpy as np
import pickle
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

class myData:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels

def distance(grid, sample):
    return np.sum((grid - sample)**2, axis=2)

class SOM:
    def __init__(self, inputs, grid_dim, max_learning_rate = 0.2, min_learning_rate = 0.01, radious = 15): # input should be flattened
        np.random.seed(10)
        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        self.grid_dim = grid_dim
        self.grid = np.random.randn(grid_dim, grid_dim, mean.shape[0])*std*0.1 + mean
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
            for candidate_x, candidate_y in np.ndindex(self.grid_dim, self.grid_dim):
                distance_from_winner = (candidate_x - win_x)**2 + (candidate_y - win_y)**2
                if distance_from_winner <= current_radious**2:
                    self.grid[candidate_x, candidate_y] += lr*(sample - self.grid[candidate_x, candidate_y])*np.exp(-distance_from_winner / 2*current_radious**2)
            
    def train(self, inputs, epochs, batch_size):
        tau = epochs**2/(np.log(2*self.radious))
        for epoch in tqdm(range(epochs)): 
            current_radious = int(self.radious*np.exp(-(epoch)**2/tau))
            lr = self.max_learning_rate - (self.max_learning_rate - self.min_learning_rate)*epoch/epochs
            for i in tqdm(range(0, inputs.shape[0], batch_size)):
                batch = inputs[i:i+batch_size]
                self.update_grid(batch, lr, current_radious)
                
                
path_for_soms = r'/home/theodor/Git_Projects/DD2424-Deep_Learning-COVID-Project/pytorch_example/dataset/soms/balanched_data.pickle'
soms_dataset = pickle.load(open(path_for_soms, 'rb'))
indexes = np.random.permutation(soms_dataset.labels.shape[0])
soms_dataset.data = soms_dataset.data[indexes]
soms_dataset.labels = soms_dataset.labels[indexes]

def reshapeImages(images):
    # setting dim of the resize
    height = 32
    width = 32
    dim = (width, height)
    # reshaping
    reshaped = [cv2.resize(np.array(image), dim, interpolation=cv2.INTER_LINEAR) for image in images]

    return np.array(reshaped)

resized_images = reshapeImages(soms_dataset.data)
flattened = resized_images.reshape(resized_images.shape[0], -1)

# Train SOM
grid_dim = 20
epochs = 1
batch_size = 75
som = SOM(flattened, grid_dim)
som.train(flattened, epochs, batch_size)

# Plot the clusters, This is going to take much time
ranking = np.zeros((grid_dim, grid_dim, 15))
for sample, label in tqdm(zip(flattened, soms_dataset.labels)):
    pos_x, pos_y = som.find_winner([sample])[0]
    ranking[int(pos_x), int(pos_y), label] += 1
ranking_winners = np.array(np.argmax(ranking, axis=2), dtype=int)
number_of_points = np.sum(ranking, axis=2)
percentage_of_winner = np.max(ranking, axis=2)/number_of_points
number_of_points /= np.sum(number_of_points)

plt.figure(figsize=(8,8))
for i, j in np.ndindex(grid_dim, grid_dim):
    plt.scatter(i,j, c='g', alpha=0.5, s=number_of_points[i,j]*900)
    plt.scatter(i,j, c='b', alpha=0.5, s=percentage_of_winner[i,j]*40)
    plt.text(i, j, ranking_winners[i,j], ha='center', va='center', 
        bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
