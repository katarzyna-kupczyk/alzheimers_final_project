import tensorflow as tf
from tensorflow.keras.optimizers import Adam

### MODEL PARAMETERS ###
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 100
INPUT_SHAPE = (224, 224, 3)
METRICS = ['AUC', 'accuracy', 'Recall', 'Precision']
OPTIMIZER = Adam(learning_rate=0.001)



### GCP DATA STORAGE ###

BUCKET_NAME = 'alzheimers-project-699'
BUCKET_TRAIN_DATA_PATH = 'data/AlzheimersDataset/train'
STORAGE_LOCATION = 'models/model.joblib'
