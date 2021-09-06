import numpy as np
import tensorflow as tf

def preprocessing(img, label):
    img = img / 255
    return img, label

def augment(img, label):
    img = tf.image.random_brightness(img, 0.2, seed=123)

    img = tf.image.stateless_random_contrast(img, 0.2, 0.5, seed=(1,2))

    return img, label
