from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import joblib

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
    return {'Greeting': 'Welcome to the Alzheimer\'s Classification API'}

@app.get('/predict')
def predict(img_path):
    img = Image.open(img_path).convert('L').resize((224, 224), Image.ANTIALIAS)
    img = np.array(img)
    img = img/255

    model = joblib.load('model.joblib')

    prediction = model.predict(img)
    print(prediction)
    return {'prediction': prediction[0]}
