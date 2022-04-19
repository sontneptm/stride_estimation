from bdb import effective
from cgi import test
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob

def load_data():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    data_list = []
    data_path_list = glob('./stride_lab_data/processed_data/*/*')
    for path in data_path_list:
        data = pd.read_csv(path).to_numpy()
        data_list.append(data)

    for i in range(len(data_list)):
        data = data_list[i]

        if i<5:
            for d in data:
                train_y.append(d[0])
                train_x.append(d[1:])
        elif i>=5:
            for d in data:
                test_y.append(d[0])
                test_x.append(d[1:])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(units=30,
        return_sequences=True,
        input_shape=[120,1])
    ])

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data()
    train_x = train_x.reshape(-1,120,1)
    test_x = test_x.reshape(-1,120,1)

    print(train_x[0])
    print(test_x)

