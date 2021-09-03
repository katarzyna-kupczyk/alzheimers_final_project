import tensorflow as tf
import joblib
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alzheimers_final_project.data import train_data_loading, test_data_loading
from alzheimers_final_project.params import STORAGE_LOCATION, BUCKET_NAME
from alzheimers_final_project.model import build_compile_model
from alzheimers_final_project.preprocess import preprocessing
from google.cloud import storage


class Trainer():
    def __init__(self):
        self.model = None
        self.train_path = None
        self.test_path = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

### MODEL PIPELINE ###
    def set_data(self):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        train_path = os.path.join(parent_path, 'raw_data/AlzheimersDataset/train')
        test_path = os.path.join(parent_path, 'raw_data/AlzheimersDataset/test')
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        # Get and preprocess data
        self.train_generator, self.validation_generator = train_data_loading(self.train_path)
        self.test_generator = test_data_loading(self.test_path)

    def set_model(self):
        # Autotune the process
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.train_generator = self.train_generator.map(preprocessing, num_parallel_calls=AUTOTUNE)
        self.validation_generator = self.validation_generator.map(preprocessing, num_parallel_calls=AUTOTUNE)
        self.test_generator = self.test_generator.map(preprocessing, num_parallel_calls=AUTOTUNE)

        self.train_generator = self.train_generator.cache().prefetch(buffer_size=AUTOTUNE)
        self.validation_generator = self.validation_generator.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_generator = self.test_generator.cache().prefetch(buffer_size=AUTOTUNE)

        self.model = build_compile_model()
        return self.model

    def fit_model(self):
        if self.model == None:
            self.set_model()
        es = EarlyStopping(patience=10, restore_best_weights=True)
        rop = ReduceLROnPlateau(monitor='val_loss', factor=0.005, patience=10, min_lr=0.005)
        self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=50, callbacks=[es, rop], verbose=1)


    def evaluate(self):
        """evaluates the model on test set and returns the Loss, AUC, Accuracy, Recall and Precision"""
        test_scores = self.model.evaluate(self.test_generator)
        scores_dict = {'Loss': test_scores[0],
                       'AUC': test_scores[1],
                       'Accuracy': test_scores[2],
                       'Recall': test_scores[3],
                       'Precision': test_scores[4]}
        return scores_dict


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

    t.set_data()
    t.load_data()
    t.set_model()
    t.fit_model()
    t.evaluate()

    # Train model and save to gcp
