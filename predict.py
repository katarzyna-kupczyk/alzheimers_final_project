from PIL import Image
import tensorflow as tf


def predict_img(path_to_prediction_data):
    image = Image.open(path_to_prediction_data).convert('RGB')
    image_array  = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image_array, (224, 224))
    image = image / 255
    image = tf.expand_dims(image, axis = 0)

    model = tf.keras.models.load_model('alz_model')
    prediction = model.predict(image)
    return {'prediction': prediction[0]}
