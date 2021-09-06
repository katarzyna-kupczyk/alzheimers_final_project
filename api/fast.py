from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {'Greeting': 'Welcome to the Alzheimer\'s Stages Classification API'}

@app.get('/predict')
def predict_img(path_to_prediction_data):
    image = Image.open(path_to_prediction_data).convert('RGB')
    image_array  = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image_array, (224, 224))
    image = image / 255
    image = tf.expand_dims(image, axis = 0)

    model = tf.keras.models.load_model('alz_model')
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
