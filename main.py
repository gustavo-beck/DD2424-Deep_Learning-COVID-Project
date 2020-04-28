# Import packages and methods
import preprocessing as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import h5py

try:
    hf = h5py.File('pre-processed-dataset\data.h5', 'r')
    print(hf.keys())
    covid = hf.get('covid-pre-processed')
    x_ray = hf.get('x_ray-pre-processed')
    print("Found data")

except IOError :
    print("Couldn't find data sets")
    # Read COVID data set
    hf = h5py.File('pre-processed-dataset\data.h5', 'w')
    covid_path = "covid-chestxray-dataset-master"
    image_covid_folder = os.path.join(covid_path, "images")
    image_list = os.listdir(image_covid_folder)
    images_PIL = [Image.open(os.path.join(image_covid_folder, image_name)) for image_name in image_list]
    covid = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True)
    hf.create_dataset('covid-pre-processed', data=covid)
    print('Saved covid pickles')

    # Read X-Ray data set
    x_ray_path = "x-ray-dataset"
    image_x_ray_folder = os.path.join(x_ray_path, "images")
    image_x_ray_list = os.listdir(image_x_ray_folder)
    images_PIL = [Image.open(os.path.join(image_x_ray_folder, image_name)) for image_name in image_x_ray_list]
    x_ray = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True)
    hf.create_dataset('x_ray-pre-processed', data=x_ray)
    print('Saved x-rays pickles')
    hf.close()


print('Cleaned Data')
plt.imshow(x_ray[32],cmap='gray')
plt.show()

print('Closing HDf5 file')
hf.close()