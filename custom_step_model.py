import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization as BN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from glob import glob

input_size = None

def setup_gpu():
    print("==== setting up GPU ====")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

def scale_data(train_data, test_data):
    print("==== scaling DATA ====")
    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data

def load_data():
    print("==== loading DATA ====")
    data_path_list = glob('./stride_lab_data/processed_data/*/*')

    data_list=None

    for path in data_path_list:
        data = pd.read_csv(path).to_numpy()
        if data_list is None: 
            data_list = data
        else: 
            data_list = np.concatenate((data_list,data), axis=0)

    x_data = data_list[:,3:]
    y_data = data_list[:,2:3]

    train_x, test_x, train_y, test_y =  train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    train_x, test_x = scale_data(train_x, test_x)

    print("train x shape: ", train_x.shape)
    print("train y shape: ", train_y.shape)
    print("test x shape: ", test_x.shape)
    print("test y shape: ", test_y.shape)

    global input_size
    input_size = len(train_x[0])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    
    return train_dataset, test_dataset

def build_cnn_model(input_size):
    print("==== building DATA ====")
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', input_shape=[input_size,1]))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same'))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same'))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same'))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4096, activation='swish'))
    model.add(BN())
    model.add(Dense(4096, activation='swish'))
    model.add(BN())
    model.add(Dense(4096, activation='swish'))
    model.add(BN())
    model.add(Dense(4096, activation='swish'))
    model.add(BN())
    model.add(Dense(1, activation=None))

    model.summary()

    return model

if __name__ == '__main__':
    setup_gpu()
    train_dataset, test_dataset = load_data()
    l_step_model = build_cnn_model(input_size)

