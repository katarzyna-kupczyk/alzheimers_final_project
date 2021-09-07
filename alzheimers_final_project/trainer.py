import tensorflow as tf
import os
import pickle
import sys
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alzheimers_final_project.data import train_data_loading, test_data_loading
from alzheimers_final_project.model import build_compile_model
from alzheimers_final_project.preprocess import preprocessing, augment

sys.setrecursionlimit(10000)

class Trainer():
    def __init__(self):
        self.model = None
        self.train_path = None
        self.test_path = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.scores_dict = None

### MODEL PIPELINE ###
    def set_data(self):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        train_path = os.path.join(parent_path, 'raw_data/AlzheimersDataset/train')
        test_path = os.path.join(parent_path, 'raw_data/AlzheimersDataset/test')
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        # Get and preprocess data
        self.train_generator, self.validation_generator = train_data_loading('/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/train')
        self.test_generator = test_data_loading('/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/test')


    def set_model(self):
        # Autotune the process
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.train_generator = self.train_generator.map(preprocessing, num_parallel_calls=AUTOTUNE)
        self.train_generator = self.train_generator.map(augment, num_parallel_calls=AUTOTUNE)
        self.validation_generator = self.validation_generator.map(preprocessing, num_parallel_calls=AUTOTUNE)
        self.test_generator = self.test_generator.map(preprocessing, num_parallel_calls=AUTOTUNE)

        self.train_generator = self.train_generator.cache().prefetch(buffer_size=AUTOTUNE)
        self.validation_generator = self.validation_generator.prefetch(buffer_size=AUTOTUNE)
        self.test_generator = self.test_generator.prefetch(buffer_size=AUTOTUNE)

        self.model = build_compile_model()
        return self.model

    def fit_model(self):
        if self.model == None:
            self.set_model()
        es = EarlyStopping(patience=10, restore_best_weights=True)
        rop = ReduceLROnPlateau(monitor='val_auc', factor=0.01, patience=5, min_lr=0.00001)
        self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=200, callbacks=[es, rop], verbose=1)


    def evaluate(self):
        """evaluates the model on test set and returns the Loss, AUC, Accuracy, Recall and Precision"""
        test_scores = self.model.evaluate(self.test_generator)
        self.scores_dict = {'Loss': test_scores[0],
                       'AUC': test_scores[1],
                       'Accuracy': test_scores[2],
                       'Recall': test_scores[3],
                       'Precision': test_scores[4]}
        return self.scores_dict


### SAVE MODEL ###
    def save_model(self, model_name):
        """ Save the trained model into folder """
        self.model.save(model_name)
        if self.scores_dict != None:
          with open(f'{model_name}_scores.pickle', 'wb') as handle:
            pickle.dump(self.scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_model(self, path_to_model):
        self.model = tf.keras.models.load_model(path_to_model)


if __name__ == "__main__":

    t = Trainer()

    t.set_data()
    t.load_data()
    t.set_model()
    t.fit_model()
    t.evaluate()

    # Train model and save
    t.save_model()
