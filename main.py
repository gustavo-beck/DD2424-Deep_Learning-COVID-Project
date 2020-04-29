# Import packages and methods
import preprocessing as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle

try:
    covid = pickle.load(open('pre-processed-dataset/covid-pre-processed.pickle', 'rb'))
    x_ray = pickle.load(open('pre-processed-dataset/x_ray-pre-processed.pickle', 'rb'))
    print("Found data")

except IOError :
    print("Couldn't find data sets")
    # Read COVID data set
    covid_path = "covid-chestxray-dataset-master"
    image_covid_folder = os.path.join(covid_path, "images")
    image_list = os.listdir(image_covid_folder)
    # Remove CT images and not matching images between databases
    df_covid = pd.read_csv('covid-chestxray-dataset-master/metadata.csv')
    matching_df = df_covid[df_covid['filename'].isin(image_list)]
    xray_list = matching_df[matching_df['modality'] == 'X-ray']['filename'].tolist()
    images_PIL = [Image.open(os.path.join(image_covid_folder, image_name)) for image_name in xray_list]
    covid = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True, covid = True)
    pickle.dump(covid, open('pre-processed-dataset/covid-pre-processed.pickle', 'wb'))
    print('Saved covid pickles')

    # Read X-Ray data set
    x_ray_path = "x-ray-dataset"
    image_x_ray_folder = os.path.join(x_ray_path, "images")
    image_x_ray_list = os.listdir(image_x_ray_folder)
    images_PIL = [Image.open(os.path.join(image_x_ray_folder, image_name)) for image_name in image_x_ray_list]
    x_ray = pp.preprocess(images_PIL, gray_scale = True, denoise = True, clahe = True, xray = True)
    pickle.dump(x_ray, open('pre-processed-dataset/x_ray-pre-processed.pickle', 'wb'))
    print('Saved x-rays pickles')


print('Cleaned Data')
plt.imshow(x_ray[32],cmap='gray')
plt.show()

print('Closing HDf5 file')