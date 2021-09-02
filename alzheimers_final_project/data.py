# Import dependencies
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Function will return the data imported, with X normalised & augmented and the target encoded...
# Need to pass in the parameter path_to_train_data with pathway on local machine!!!
# Variables returned are train_generator, validation_generator

# Insert your local path to train data and test data here:
path_to_train_data = '/raw_data/ALzheimersDataset/train'
path_to_test_data = '/raw_data/ALzheimersDataset/test'

def train_data_loading_preprocessing(path_to_train_data):
    datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 5,
        zoom_range = (0.1, 0.1),
        data_format = 'channels_last',
        validation_split = 0.2,
        dtype = tf.bfloat16)

    # Train Generator
    train_generator = datagen.flow_from_directory(
    path_to_train_data,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    subset = 'training',
    seed = 123)

    # Validation Generator
    validation_generator = datagen.flow_from_directory(
    path_to_train_data,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    subset = 'validation',
    seed = 123)

    return train_generator, validation_generator

def test_data_loading_preprocessing(path_to_test_data):
    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        dtype = tf.bfloat16)

    test_generator = test_datagen.flow_from_directory(
        path_to_test_data,
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = True,
        seed=123)

    return test_generator


if __name__ == '__main__':
    train_generator, validation_generator = train_data_loading_preprocessing(path_to_train_data)
    test_generator = test_data_loading_preprocessing(path_to_test_data)
