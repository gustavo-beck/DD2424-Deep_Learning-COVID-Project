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
    def __init__(self, model_path, architecture, device):
        # Load model
        if architecture == 'Dense121':
            model= torch.load(model_path)
            model.to(device)
            self.model = model.module.densenet121.features
        self.model.eval() # Doesn't store gradients

        # Load penultimate weight's layer
        if architecture == 'Dense121':
            self.weights = list(self.model.parameters())[-2]

        # Prepare image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.transform = transforms.Compose(image_transform)

    def generateMap(self, pathImageFile, pathOutputFile):
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transform(imageData)

        # Walk the image through the model
        output = self.model(imageData)

        # Generate heatmap
        heatmap = None
        for i in range (len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map

        #---- Blend original and heatmap 
        npHeatmap = heatmap.data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1).convert('RGB')
        height, width, channels = imgOriginal.shape
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (height,width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)

pathInputImage = 'CHANGETHISSHIT/00009285_000.png'
pathOutputImage = 'CHANGETHISSHIT/heatmap.png'
pathModel = 'CHANGETHISSHIT'

architecture = 'DENSE-NET-121'

transCrop = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
h = XrayHeatmap(pathModel, architecture, device)
h.generate(pathInputImage, pathOutputImage)
