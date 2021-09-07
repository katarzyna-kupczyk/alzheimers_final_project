from PIL import Image
import tensorflow as tf
import numpy as np
import sys

sys.setrecursionlimit(10000)

path_to_img = "/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/test/MildDemented/26 (19).jpg"

def predict_img(path_to_prediction_data):
    image = Image.open(path_to_prediction_data).convert('RGB')
    image_array  = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image_array, (224, 224))
    image = image / 255
    image = tf.expand_dims(image, axis = 0)
    print(image)
    model = tf.keras.models.load_model('alz_model_h5.h5')
    print(model)
    prediction = model.predict(image)
    prediction_array = prediction[0]
    max_value = np.max(prediction_array)
    if max_value == prediction_array[0]:
        classification = 'Mild Demented'
    elif max_value == prediction_array[1]:
        classification = 'Moderate Demented'
    elif max_value == prediction_array[2]:
        classification = 'Non Demented'
    else:
        classification = 'Very Mild Demented'

    return {'prediction': classification}

predict_img(path_to_img)
