# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow import expand_dims
from tensorflow.keras.callbacks import EarlyStopping

# Function will return the data imported, with X normalised & augmented and the target encoded...
# Need to pass in the parameter path_to_train_data with pathway on local machine!!!
# Variables returned are train_generator, validation_generator

def data_preprocessing(path_to_train_data='/Users/LG/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/train'):
    datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 5,
        zoom_range = (0.90, 0.90),
        brightness_range = (0.95, 0.95),
        horizontal_flip = True,
        vertical_flip = True,
        data_format = 'channels_last',
        validation_split = 0.2,
        dtype = float
    )
    # Train Generator
    train_generator = datagen.flow_from_directory(
    '/Users/LG/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/train',
    target_size = (224, 224),
    batch_size = (32),
    class_mode = 'categorical',
    shuffle = True,
    subset = 'training',
    seed = 123
    )
    # Validation Generator
    validation_generator = datagen.flow_from_directory(
    '/Users/LG/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/train',
    target_size = (224, 224),
    batch_size = (32),
    class_mode = 'categorical',
    shuffle = True,
    subset = 'validation',
    seed = 123
    )
    return train_generator, validation_generator


# Example call
# train_generator, validation_generator = data_preprocessing()
