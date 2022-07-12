import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, BatchNormalization as BN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from glob import glob

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)

def load_data(mode="loso"):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    x_data =[]
    y_data =[]
    data_list = []
    data_path_list = glob('./stride_lab_data/processed_data/*/*')

    for path in data_path_list:
        print(path)
        data = pd.read_csv(path).to_numpy()
        data_list.append(data)

    if mode == "loso":
        div = 10

        for i in range(len(data_list)):
            data = data_list[i]

            if i<div:
                for d in data:
                    train_y.append(d[0])
                    train_x.append(d[4:])
            elif i>=div:
                for d in data:
                    test_y.append(d[0])
                    test_x.append(d[4:])

    elif mode=="shuffle":
        for i in range(len(data_list)):
            data = data_list[i]

            for d in data:
                y_data.append(d[:2])
                x_data.append(d[4:])

        train_x, test_x, train_y, test_y =  train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def build_conv1_model(LR, INPUT_SIZE):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='swish', input_shape=[INPUT_SIZE,1]))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='swish'))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='swish'))
    model.add(BN())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='swish'))
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
    model.add(Dense(2, activation=None))
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')

    model.summary()

    return model

if __name__ == "__main__":
    setup_gpu()
    train_x, train_y, test_x, test_y = load_data(mode="shuffle")

    scaler = StandardScaler()

    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    INPUT_SIZE = len(train_x[0])
    EPOCH = 2000
    BATCH_SIZE = 32
    LR = 1.46e-4

    train_x = train_x.reshape(-1,INPUT_SIZE,1)
    test_x = test_x.reshape(-1,INPUT_SIZE,1)

    step_model = build_conv1_model(LR, INPUT_SIZE)

    #plot_model(step_model, to_file='model.png', show_shapes=True, show_layer_names=True)

    step_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, shuffle=True, validation_split=0.1)

    predict_y = step_model.predict(test_x)

    test_y = test_y.squeeze()

    predict_y = predict_y.squeeze()

    print("======= report ==========")
    print(test_y)
    print(predict_y)

    print("MAE: ",mean_absolute_error(y_true=test_y, y_pred=predict_y))
    print("=========================")

    print("ME: ", np.mean(np.subtract(test_y, predict_y)))
    print("std: ", np.std(np.subtract(test_y, predict_y)))
    print("=========================")

    print("relative error: ", np.mean(np.divide(np.absolute(np.subtract(test_y, predict_y)), test_y))*100)
    print("=========================")
    
    print("r2 score: ", r2_score(y_true=test_y, y_pred=predict_y))
    print("=========================")