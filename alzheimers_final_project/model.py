from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.densenet import DenseNet121
from alzheimers_final_project.params import INPUT_SHAPE, METRICS, OPTIMIZER

def build_compile_model():

    # DenseNet121 Base Model
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    base_model.trainable = False

    # Base Model + Trainable Layers
    model = Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=METRICS)

    return model
