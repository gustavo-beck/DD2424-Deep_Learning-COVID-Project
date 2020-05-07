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
print('libraries alright')
path="/home/theodor/Desktop/dataset/"
img_name=[]
label=[]
for root, directories, files in os.walk(path):
    for file in files:
        img_name.append(root+"/"+file)
        label.append(root.split("/")[-1])

# we define all the transformations we want to do with the images
# RandomAffine does a serries of transformations as explained here https://www.mathworks.com/discovery/affine-transformation.html
# with those transformations, the network will be able to handle distorted input
# torchvision.transforms.ToTensor() converts image to tensor input
images_size = 224 # 224
train_Aug = torchvision.transforms.Compose([torchvision.transforms.Resize((images_size, images_size)), torchvision.transforms.RandomRotation((-20, 20)), torchvision.transforms.RandomAffine(0, translate=None, scale=[0.7, 1.3], shear=None, resample=False, fillcolor=0), torchvision.transforms.ToTensor()])
test_Aug = torchvision.transforms.Compose([torchvision.transforms.Resize((images_size, images_size)), torchvision.transforms.ToTensor()]) # validation shouldn't be distorted, only resized

# not neccessary to define it, but it helps organize the input
# class is a child of torch.utils.data.Dataset
# getitem enables when we do CustomDatasetFromImages[index]
class CustomDatasetFromImages(torch.utils.data.Dataset):
    def __init__(self, img_name, label, transforms=None): 
        self.image_arr = np.asarray(img_name)
        self.label_arr = np.asarray(label)
        self.data_len = len(img_name)
        self.transforms = transforms
    def __getitem__(self, index):
        single_img_name = self.image_arr[index]
        img_array = Image.open(single_img_name).convert('RGB')
        if self.transforms is not None:
            img_array = self.transforms(img_array)
            image_label = self.label_arr[index]
        return (single_img_name, img_array, image_label)
    def __len__(self):
        return self.data_len

# shuffling the samples
index = np.random.permutation(len(img_name))
img_name = np.array(img_name)[index]
label = np.array(label)[index]
# labels to 0 and 1, pytorch doen't work with onehot but with class numbers
new_labels = np.zeros((len(label)), dtype=int) # has to be int and 1D-vector or else we will have problems
new_labels[label == 'covid'] = 1
label = new_labels
# splitting to train-test
train_nums = 25
train_set = img_name[:train_nums]
train_labels= label[:train_nums]
test_set = img_name[train_nums:]
test_labels= label[train_nums:]
train_set = CustomDatasetFromImages(train_set, train_labels, transforms=train_Aug)
test_set = CustomDatasetFromImages(test_set, test_labels, transforms=test_Aug)
# The loader is used to slpit the input and validation to batches, it returns arrays with the input in batches
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=2,shuffle=True) 
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, num_workers=2, shuffle=False)
print(train_set[0][0], train_set[0][1].sum(), train_set[0][2])
print(train_set[0][0], train_set[0][1].sum(), train_set[0][2])



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)) # 2x2 --> 1 2h->1h and 2w->1w : 28x28 --> 14x14
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)) #2x2 --> 1 2h->1h and 2w->1w : 14x14 --> 7x7
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(14 * 14 * 32, 2)
        #self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        ## vectorizing the input matrix
        layer2_vec = layer2.view(-1, 32*14*14)
        return torch.nn.Softmax(1)(self.fc1(self.drop_out(layer2_vec)))

model = torchvision.models.resnet18(pretrained=True)
# model.connection1 = torch.nn.Linear(512, 220)

model.fc = torch.nn.Linear(512, 2) # fc stands for fully connected layer 512 input 2 the output
model = torch.nn.Sequential(model, torch.nn.Softmax(1)) # We do a sequential pipeline
#test_model = torch.nn.Sequential(torch.nn.Conv2d(1, 20, 5), )
cov_model = ConvNet()
model = cov_model
criterion = torch.nn.CrossEntropyLoss()
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
    for link, batch_images, batch_labels in trainloader:
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

