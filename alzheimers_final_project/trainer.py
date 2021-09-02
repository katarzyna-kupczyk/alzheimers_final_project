import tensorflow as tf
import joblib
import os
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from alzheimers_final_project.data import train_data_loading_preprocessing, test_data_loading_preprocessing
from alzheimers_final_project.params import STORAGE_LOCATION, BUCKET_NAME
from google.cloud import storage



class Trainer():
    def __init__(self, local=True):
        self.pipeline = None
        self.local = local
        self.path = None

### MODEL PIPELINE ###
    def set_data(self):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(parent_path, 'raw_data/AlzheimersDataset')
        self.path = path

    def set_pipeline(self):
        # Get and preprocess data
        train_generator, validation_generator = train_data_loading_preprocessing(self.path)
        X_test = test_data_loading_preprocessing

        # Autotune the process
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_generator = train_generator.cache().prefetch(buffer_size=AUTOTUNE)
        validation_generator = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)




        self.pipeline = ...

        return self.pipeline

### SAVE MODEL TO GCP ###
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == "__main__":
    t = Trainer()
    #set_pipeline
    # t.set_pipeline()
    t.set_data()

    # Train model and save to gcp
