import numpy as np
import tensorflow as tf

def preprocessing(img, label):
    img = img / 255,
    return img, label

# def augmentation(img, label):
#     if np.random.rand(1) < 0.15:
#         ...
#     elif np.random.rand(1) < 0.15:
#         ...
#     return img, label
