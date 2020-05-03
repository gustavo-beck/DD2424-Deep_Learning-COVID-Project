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
        single_img_name = 2
        return (single_img_name, img_array, self.labels_dict[image_label])
    def __len__(self):
        return self.data_len

# shuffling the samples
train_set = CustomDatasetFromImages(directories['train'], class_dict, transforms=our_transforms) # train[0], train[1], .. , train[15]
val_set = CustomDatasetFromImages(directories['val'], class_dict, transforms=our_transforms)
test_set = CustomDatasetFromImages(directories['test'], class_dict, transforms=our_transforms)
# The loader is used to slpit the input and validation to batches, it returns arrays with the input in batches
trainloader = torch.utils.data.DataLoader(train_set, batch_size=250, num_workers=2,shuffle=False) 
testloader = torch.utils.data.DataLoader(test_set, batch_size=250, num_workers=2, shuffle=False)

print('Calculating frequencies..')
freq = np.zeros(len(class_dict.keys()))
for sample in train_set:
    freq[sample[2]]+=1
plt.bar(np.arange(0,15,1), freq)
plt.xticks(np.arange(0,15,1), class_dict.keys())
plt.show()
weights = len(train_set)/freq/len(class_dict.keys())
class_weights = torch.FloatTensor(weights)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)) # 2x2 --> 1 2h->1h and 2w->1w : 28x28 --> 14x14
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)) #2x2 --> 1 2h->1h and 2w->1w : 14x14 --> 7x7
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(14 * 14 * 32, 15)
        #self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        ## vectorizing the input matrix
        layer2_vec = layer2.view(-1, 32*14*14)
        return torch.nn.Softmax(1)(self.fc1(self.drop_out(layer2_vec)))

model = torchvision.models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.connection1 = torch.nn.Linear(512, 220)

model.fc = torch.nn.Linear(512, 15) # fc stands for fully connected layer 512 input 2 the output
model = torch.nn.Sequential(model, torch.nn.Softmax(1)) # We do a sequential pipeline
#test_model = torch.nn.Sequential(torch.nn.Conv2d(1, 20, 5), )
cov_model = ConvNet()
model = cov_model
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
# scheduler is used to adjust the learning rate,
# this oparticular scheduler uses the validation accuracy to adjust the learning rate,
# other schedulers dont require the validation accuracy
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-06)
# in pytorch training isn't that easy as keras, but we have additional flexibility
epochs = 10
for epoch in range(epochs):
    print(epoch)
    training_acc = 0
    training_samples = 0
    for link, batch_images, batch_labels in tqdm(trainloader):
        optimizer.zero_grad() # we set the gradients to zero
        outputs = model(batch_images) # feed forward
        _, pred = torch.max(outputs.data, 1) # max returns the maximum value and the argmax, 1 is the axis
        training_acc += (pred == batch_labels).sum().item() # we sum all the the correctly classified samples, item() coverts the 1d tensor to a number
        training_samples  += batch_labels.size(0) # counting the samples
        loss = criterion(outputs, batch_labels) # claculating loss
        loss.backward() # backpropagate the loss
        optimizer.step() # updating parameters
    print('Training acc:', training_acc/training_samples)
    correct = 0
    total = 0
    for link, batch_images, batch_labels in testloader:
        outputs = model(batch_images)
        _, pred = torch.max(outputs.data, 1)
        correct += (pred == batch_labels).sum().item()
        total += batch_labels.size(0)
    correct = correct/total
    print('Test acc:', correct)
    scheduler.step(correct) # update learning rate

