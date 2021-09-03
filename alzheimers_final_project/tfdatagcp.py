from alzheimers_final_project.data import train_data_loading, test_data_loading, path_to_train_data, path_to_test_data
from alzheimers_final_project.params import BUCKET_NAME
import tensorflow as tf

def load_tf_datasets_gcp():
    train_generator, validation_generator = train_data_loading(path_to_train_data)
    test_generator = test_data_loading(path_to_test_data)


    path = '/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data'
    tf.data.experimental.save(train_generator, path, compression=None, shard_func=None)
    tf.data.experimental.save(validation_generator, path, compression=None, shard_func=None)
    tf.data.experimental.save(test_generator, path, compression=None, shard_func=None)

if __name__ == '__main__':
    load_tf_datasets_gcp()








# import tempfile
# path = os.path.join(tempfile.gettempdir(), "saved_data")
# #Save a dataset
# dataset = tf.data.Dataset.range(2)
# tf.data.experimental.save(dataset, path)
