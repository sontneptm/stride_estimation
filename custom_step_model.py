import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization as BN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from glob import glob

def setup_gpu():
    print("==== setting up GPU ====")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

class StepModel():
    def __init__(self) -> None:
        self.epochs = 1000
        self.input_size = None

        self.train_dataset =None
        self.val_dataset =None
        self.test_dataset =None

        self.load_data()
        self.build_cnn_model

    def scale_data(self, train_data, val_data, test_data):
        print("==== scaling DATA ====")
        scaler = StandardScaler()

        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

        return train_data, val_data, test_data

    def load_data(self):
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
        train_x, val_x, train_y, val_y =  train_test_split(train_x, train_y, test_size=0.2, random_state=42)

        train_x, val_x, test_x = self.scale_data(train_x, val_x, test_x)

        print("train x shape: ", train_x.shape)
        print("train y shape: ", train_y.shape)
        print("val x shape: ", val_x.shape)
        print("val y shape: ", val_y.shape)
        print("test x shape: ", test_x.shape)
        print("test y shape: ", test_y.shape)

        self.input_size = len(train_x[0])

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        self.val_dataset = self.val_dataset.shuffle(buffer_size=1024).batch(batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    def build_cnn_model(self):
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

    def setup_optimizers(self):
        l_step_optimizer = Adam(learning_rate=learning_rate)
        l_step_loss = MSE()

        return l_step_optimizer, l_step_loss

    def train(self):
        for epoch in range(epochs):
            print("epoch " , epoch)

            with tf.GradientTape as tape:
                logits = model()


if __name__ == '__main__':
    setup_gpu()
    # train_dataset, val_dataset, test_dataset = load_data()

    # l_step_model = build_cnn_model(input_size)
    # l_step_optimizer, l_step_loss = setup_optimizers()

    # setup_optimizers(1.46e-4)
    pass
