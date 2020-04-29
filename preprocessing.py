##### Import Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

##### Pre-Processing #####
def display(a, b, c, d, title1 = "Original", title2 = "Gray", title3 = "Denoised", title4 = "Clahe"):
    plt.figure(figsize=(10,10))
    plt.subplot(141), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(b,cmap='gray'), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(c,cmap='gray'), plt.title(title3)
    plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(d,cmap='gray'), plt.title(title4)
    plt.xticks([]), plt.yticks([])
    plt.show()

def reshapeImages(images):

    # setting dim of the resize
    height = 512
    width = 512
    dim = (width, height)

    # reshaping
    reshaped = [cv2.resize(np.array(image), dim, interpolation=cv2.INTER_LINEAR) for image in images]

    return reshaped

def grayScaler(images):
    # Setting gray scale
    gray_images = [image.convert('L') for image in images]
    return gray_images

def denoiser(images):
    blurred_images = [cv2.GaussianBlur(image, (5, 5), 0) for image in images]
    return blurred_images

def contrastLAHE(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_clahe = [clahe.apply(image) for image in images]
    return img_clahe

def preprocess(images, gray_scale = False, denoise = False, clahe = False, covid = False, xray = False):
    img_o = images[32]
    # Transform to gray scale
    if gray_scale:
        images_gray = grayScaler(images)
        img_g = images_gray[32]
        print("Images transformed to Gray Scale")

    # Reshape images
    images_reshaped = reshapeImages(images_gray)
    print("Images reshaped to 512x512")

    # Add blur
    if denoise:
        images_denoised = denoiser(images_reshaped)
        img_d = images_denoised[32]
        print("Images Blured")
    
    # Contrast Limited Adaptive Histogram Equalization
    if clahe:
        images_clahe = contrastLAHE(images_denoised)
        img_c = images_clahe[32]
        print("Images applied Contrast Limited Adaptive Histogram Equalization")

    display(img_o, img_g, img_d, img_c)
    return images_clahe




