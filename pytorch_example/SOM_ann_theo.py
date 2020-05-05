#################################################################################
import os
import pylab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, Counter

from tqdm import tqdm
from scipy import stats

def dist(vector): # Euclidean distance of a vector
    d = np.zeros((vector.shape[0],1))
    for i in range(vector.shape[0]):
        d[i,0] = np.sqrt(np.sum(vector[i,:]**2))
    return d

# SELF ORGANIZED MAP

class SOMDataClustering:
    def __init__(self,
                 fdim=(349,31),
                 weight_shape=(100,31),
                 neighbourhood_0=50,
                 step_learn=0.1,
                 n_epochs=100):

        self.input = np.loadtxt('data_lab2/votes.dat', dtype=float, delimiter=',').reshape(fdim)  # input dataset
        self.weight = np.random.uniform(0, 1, weight_shape)  # weights from the model
        # self.weight = np.mean(self.input, axis=0) * np.ones(weight_shape)
        self.n_patterns = self.input.shape[0]  # total number of patterns
        self.n_output_patterns = self.weight.shape[0]  # number of nodes in the output grid
        self.n_neighbours = neighbourhood_0  # current number of neighbors
        self.step_learn = step_learn  # learning rate
        self.n_epochs = n_epochs  # number of epochs
        self.node2freq = np.ones(self.n_output_patterns)  # for the frequency method

    def findRowWinnerWeight(self, cand_attr):

        distance = [0] * self.n_output_patterns
        for i in range(self.n_output_patterns):
            distance[i] = self.node2freq[i] * np.linalg.norm(self.weight[i,:] - cand_attr)

        pick_row = np.argmin(distance)
        return pick_row

    def findNeighbors(self, winner_idx):
        wn_neighbors_idx = [0] * self.n_output_patterns
        curr_neigh = 0
        stack = [winner_idx]
        visited = set()

        while curr_neigh <= self.n_neighbours and stack:
            next_idx = stack.pop()
            if next_idx in visited:
                continue

            wn_neighbors_idx[next_idx] = 1
            curr_neigh += 1
            visited.add(next_idx)
            i = next_idx // 10
            j = next_idx % 10

            # add neighbors (BFS)
            if 0 <= i - 1 < 10:
                up = (i - 1) * 10 + j
                stack.append(up)

            if 0 <= i + 1 < 10:
                down = (i + 1) * 10 + j
                stack.append(down)

            if 0 <= j - 1 < 10:
                left = i * 10 + j - 1
                stack.append(left)

            if 0 <= j + 1 < 10:
                right = i * 10 + j + 1
                stack.append(right)

        return wn_neighbors_idx

    def updateWeights(self, attr, j, wn_neighbors_idx):
        for i in range(self.n_output_patterns):
            self.weight[i] += self.step_learn * wn_neighbors_idx[i] * (attr[j] - self.weight[i])

    def train(self):  # competition + neighbourhood
        
        counter_epoch = 0
        for counter_epoch in tqdm(range(self.n_epochs), desc='Number of epochs'):

            for j in range(self.n_patterns):
                # winner node
                cand_attr = self.input[j, :]
                winner_row = self.findRowWinnerWeight(cand_attr)
                self.node2freq[winner_row] += 1

                # keep winner and neighbourhood
                wn_neighbors_idx = self.findNeighbors(winner_row)

                # update winner and neighbors
                self.updateWeights(self.input, j, wn_neighbors_idx)

            # update the number of neighbors
            if counter_epoch % 10 == 0:
                # self.plotResult()
                self.n_neighbours /= 2
                self.n_neighbours = max(self.n_neighbours, 0)


    def output(self, x):
        # use the SOM with this function
        return self.findRowWinnerWeight(x)

    def plotResult(self, file, title, index2name=None):
        category = np.loadtxt(file, dtype=int, comments='%')
        cat_counter = Counter(category)
        n_cat = len(cat_counter.keys())

        winners_idx = np.apply_along_axis(self.output, 1, self.input)
        category_symb = ['ro', 'bo', 'go', 'ko', 'co', 'yo', 'mo',
                         'rx', 'bx', 'gx', 'kx', 'cx', 'yx', 'mx']*100

        point_size_offset = 1.2
        styles = [0 for i in range(n_cat)]
        for i in range(self.n_output_patterns):
            cat_samples_for_i = category[winners_idx == i]
            if cat_samples_for_i.size > 0:
                tmp = np.bincount(cat_samples_for_i)
                most_used_category = np.argmax(tmp)
                point_size = np.max(tmp)
                col = i//10
                row = i % 10
                styles[most_used_category], = plt.plot(row, col, category_symb[most_used_category], markersize=(point_size*point_size_offset), alpha=0.7)

        if index2name:
            plt.legend(styles, [index2name[i] for i in range(n_cat)])

        plt.title(title, fontweight="bold")
        plt.show()


som = SOMDataClustering()
som.train()

cat = 'party'

if cat == 'sex':
    index2name = {0: 'Male', 1: 'Female'}
    som.plotResult('data_lab2/mpsex.dat', "Gender SOM", index2name)


if cat == 'party':
    # Social Democrats = s,
    # New Moderates = M,
    # Centre party = C,
    # Left party = V,
    # Christaind democrats = KD,
    # Liberals = fp,
    # Greeen party = mp

    # index2name = {0: 'no party', 1:'m', 2:'fp', 3:'s', 4:'v', 5:'mp', 6:'kd', 7:'c'}
    index2name = {0: 'no party', 1: 'New Moderates', 2: 'Liberals', 3: 'Social Democrats', 4: 'Left party',
                  5: 'Greeen party', 6: 'Christaind democrats', 7: 'Centre party'}
    som.plotResult('data_lab2/mpparty.dat', "Party SOM", index2name)

if cat == 'district':
    som.plotResult('data_lab2/mpdistrict.dat', "District SOM")