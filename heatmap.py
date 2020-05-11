##### IMPORT LIBRARIES #####
import os
import numpy as np
import time
import sys
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd

##### CREATE HEATMAP CLASS #####
class XrayHeatmap():
    def __init__(self, model_path, architecture, device, label):
        # Load model
        if architecture == 'Dense121':
            model= torch.load(model_path, map_location='cpu')
            model.to(device)
            self.model = model.features
        self.model.eval() # Doesn't store gradients

        # Load penultimate weight's layer
        if architecture == 'Dense121':
            self.weights = list(model.parameters())[-1][label]

        # Prepare image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def generateMap(self, pathImageFile, pathOutputFile, device):
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.image_transform(imageData)
        imageData = imageData.unsqueeze_(0)
        imageData = imageData.to(device)

        # Walk the image through the model
        output = self.model(imageData)
        output = torch.nn.functional.relu(output)

        # Generate heatmap
        heatmap = None
        for i in range (len(self.weights)):
            map = output[0, i, :, :]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map

        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()
        npHeatmap = np.abs(npHeatmap)

        imgOriginal = cv2.imread(pathImageFile, 1)
        height, width, channels = imgOriginal.shape
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.7 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)

df = pd.read_csv("organized_dataset.csv")
df.pop('No Finding')  # Deleting the column no findings
cols = np.array(df.columns)
image = ['00000001_000.png',
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
pathModel = 'final_model_39.pt'
architecture = 'Dense121'
transCrop = 224
for i in range(len(image)):
    disease = np.argmax((df[df['Image Index'] == image[i]] == 1) *1)
    pathInputImage = 'images_small/' + image[i]
    pathOutputImage = 'heatmap_images/heatmap_' + image[i] + '_' + cols[disease] + '.png'
    label = disease - 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h = XrayHeatmap(pathModel, architecture, device, label)
    h.generateMap(pathInputImage, pathOutputImage, device)
