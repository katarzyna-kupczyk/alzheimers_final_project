import tensorflow as tf

### MODEL PARAMETERS ###
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 100
INPUT_SHAPE = (224, 224, 3)
METRICS = ['AUC', 'accuracy', 'Recall', 'Precision']



### GCP DATA STORAGE ###

BUCKET_NAME = 'alzheimers-project-699'
BUCKET_TRAIN_DATA_PATH = 'data/AlzheimersDataset/train'
STORAGE_LOCATION = 'models/model.joblib'
