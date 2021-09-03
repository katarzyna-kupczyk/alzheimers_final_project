# Import dependencies
from tensorflow.keras.preprocessing import image_dataset_from_directory
from alzheimers_final_project.params import BATCH_SIZE, IMAGE_SIZE


# Insert your local path to train data and test data here:
path_to_train_data = '/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/train'
path_to_test_data = '/Users/katarzynakupczyk/code/katarzyna-kupczyk/alzheimers_final_project/raw_data/AlzheimersDataset/test'

def train_data_loading(path_to_train_data):

    # Train Generator
    train_generator = image_dataset_from_directory(
    path_to_train_data,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
    labels = 'inferred',
    label_mode = 'categorical',
    shuffle = True,
    validation_split = 0.2,
    subset = 'training',
    seed = 123)

    # Validation Generator
    validation_generator = image_dataset_from_directory(
    path_to_train_data,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
    labels = 'inferred',
    label_mode = 'categorical',
    shuffle = True,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123)

    return train_generator, validation_generator

def test_data_loading(path_to_test_data):

    # Test Generator
    test_generator = image_dataset_from_directory(
    path_to_test_data,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
    labels = 'inferred',
    label_mode = 'categorical',
    shuffle = True,
    seed = 123)

    return test_generator


if __name__ == '__main__':
    train_generator, validation_generator = train_data_loading(path_to_train_data)
    test_generator = test_data_loading(path_to_test_data)
