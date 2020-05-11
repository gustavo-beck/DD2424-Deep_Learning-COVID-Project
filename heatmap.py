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


##### CREATE HEATMAP CLASS #####
class XrayHeatmap():
    def __init__(self, model_path, architecture, device, label):
        # Load model
        if architecture == 'Dense121':
            model= torch.load(model_path)
            model.to(device)
            self.model = model.features
        self.model.eval() # Doesn't store gradients

        # Load penultimate weight's layer
        if architecture == 'Dense121':
            self.weights = list(model.parameters())[-1][10]

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
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)

pathInputImage = 'images_small/00000001_000.png'
pathOutputImage = 'heatmap_images/heatmap.png'
pathModel = 'final_model.pt'

architecture = 'Dense121'

transCrop = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
h = XrayHeatmap(pathModel, architecture, device, 5)
h.generateMap(pathInputImage, pathOutputImage, device)
