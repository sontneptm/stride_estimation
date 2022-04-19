import imp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
import numpy as np
import pandas as pd
from glob import glob

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
        data = pd.read_csv(path).to_numpy()
        data_list.append(data)

    if mode == "loso":
        div = 6

        for i in range(len(data_list)):
            data = data_list[i]

            if i<div:
                for d in data:
                    train_y.append(d[0])
                    train_x.append(d[1:])
            elif i>=div:
                for d in data:
                    test_y.append(d[0])
                    test_x.append(d[1:])

    elif mode=="shuffle":
        for i in range(len(data_list)):
            data = data_list[i]

            for d in data:
                y_data.append(d[0])
                x_data.append(d[1:])

        scaler = MinMaxScaler((0,1))

        x_data = scaler.fit_transform(x_data)

        train_x, test_x, train_y, test_y =  train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def build_model():
    model = Sequential()
    # model.add(Conv1D(filters=128, kernel_size=4, input_shape=[120,1]))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=128, kernel_size=4))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    model.add(LSTM(units=128, return_sequences=True, input_shape=[120,1]))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='swish'))
    model.add(Dense(1024, activation='swish'))
    model.add(Dense(1024, activation='swish'))
    model.add(Dense(1, activation='swish'))
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    model.summary()

    return model

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data(mode="shuffle")
  
    train_x = train_x.reshape(-1,120,1)
    test_x = test_x.reshape(-1,120,1)

    EPOCH = 1000
    BATCH_SIZE = 8
    LR = 1.46e-4
    
    model = build_model()
    model.fit(train_x, train_y, batch_size=16, epochs=EPOCH, shuffle=True, validation_split=(0.1))

    predict_y = model.predict(test_x)

    print(r2_score(y_true=test_y, y_pred=predict_y))

    for i in range(len(test_y)):
        print("real: " + str(test_y[i]) + " predicted: " + str(predict_y[i]))



