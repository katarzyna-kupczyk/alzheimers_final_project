from PIL import Image
import numpy as np
import joblib

def preprocessing(img_path):
    img = Image.open(img_path).convert('L').resize((224, 224), Image.ANTIALIAS)
    img = np.array(img)
    img_preproc = img/255
    return img_preproc

def predict(img_preproc, model_path):
    model = joblib.load(model_path)
    prediction = model.predict(img_preproc)
    return prediction
