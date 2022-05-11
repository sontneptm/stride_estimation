import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization as BN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.metrics import MeanSquaredError as MSE_metrics 
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from glob import glob
import time

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
        self.learning_rate = 1.46e-4
        self.batch_size = 32
        self.input_size = None

        self.train_dataset =None
        self.val_dataset =None
        self.test_dataset =None   
        self.load_dataset()

        self.l_step_optimizer = None
        self.l_step_loss = None
        self.setup_optimizers()

        self.train_acc_metric = None 
        self.val_acc_metric = None
        self.setup_metrics()

        self.l_step_model = self.build_cnn_model()

    def scale_data(self, train_data, val_data, test_data):
        print("==== scaling DATA ====")
        
        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
        scaler = RobustScaler()

        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

        return train_data, val_data, test_data

    def load_dataset(self):
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
        self.input_size = len(train_x[0])

        train_x = train_x.reshape(-1, self.input_size, 1)
        val_x = val_x.reshape(-1, self.input_size, 1)
        test_x = test_x.reshape(-1, self.input_size, 1)

        self.test_x = test_x
        self.test_y = test_y

        print("train x shape: ", train_x.shape)
        print("train y shape: ", train_y.shape)
        print("val x shape: ", val_x.shape)
        print("val y shape: ", val_y.shape)
        print("test x shape: ", test_x.shape)
        print("test y shape: ", test_y.shape)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        self.org_train_dataset = self.train_dataset
        self.train_dataset = self.train_dataset.shuffle(buffer_size=128, reshuffle_each_iteration=True).batch(self.batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        self.val_dataset = self.val_dataset.batch(self.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    def build_cnn_model(self):
        print("==== building MODEL ====")
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='swish', input_shape=[self.input_size,1]))
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
        model.add(Dense(4048, activation='swish'))
        model.add(BN())
        model.add(Dense(4048, activation='swish'))
        model.add(BN())
        model.add(Dense(4048, activation='swish'))
        model.add(BN())
        model.add(Dense(4048, activation='swish'))
        model.add(BN())
        model.add(Dense(1, activation=None))
        model.summary()

        return model

    def setup_optimizers(self):
        print("==== setting up OPT ====")
        self.l_step_optimizer = Adam(learning_rate=self.learning_rate)
        self.l_step_loss = MSE()

    def setup_metrics(self):
        print("==== setting up METRICS ====")
        self.train_acc_metric = MSE_metrics()
        self.val_acc_metric = MSE_metrics()
    
    def train(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            print("epoch " , epoch, end=" -> ")
            epoch_time = time.time()
            
            # TRAIN LOOP with BATCH
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.l_step_model(x_batch_train, training=True)
                    l_step_loss_value = self.l_step_loss(y_batch_train, logits)

                grads = tape.gradient(l_step_loss_value, self.l_step_model.trainable_variables)

                self.l_step_optimizer.apply_gradients(zip(grads, self.l_step_model.trainable_variables))

                train_acc = self.train_acc_metric.update_state(y_batch_train, logits)
                
            train_acc = self.train_acc_metric.result()

            print("Training loss : %.4f" % float(l_step_loss_value), end='\t')
            print("Training acc : %.4f" % float(train_acc), end='\t')

            self.train_acc_metric.reset_states()

            # VALIDATION LOOP with BATCH
            for x_batch_val, y_batch_val in self.val_dataset:
                val_logits = self.l_step_model(x_batch_val, training=False)
                self.val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),), end='\t')

            self.train_dataset = self.org_train_dataset.shuffle(buffer_size=128, reshuffle_each_iteration=True).batch(self.batch_size)

            print("Time taken: %.2fs" % (time.time() - epoch_time))

        print("Whole Time taken: %.2fs" % (time.time() - start_time))

    @tf.function
    def train_with_compile(self):
        l_step_loss_value = 0.0

        for epoch in range(self.epochs):
            tf.print("epoch " , epoch, end=" -> ")
            start_time = time.time()
            
            # TRAIN LOOP with BATCH
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.l_step_model(x_batch_train, training=True)
                    l_step_loss_value = self.l_step_loss(y_batch_train, logits)

                grads = tape.gradient(l_step_loss_value, self.l_step_model.trainable_weights)

                self.l_step_optimizer.apply_gradients(zip(grads, self.l_step_model.trainable_weights))

                train_acc = self.train_acc_metric.update_state(y_batch_train, logits)
                
            train_acc = self.train_acc_metric.result()

            tf.print("Training loss : ", float(l_step_loss_value), end='\t')
            tf.print("Training acc : " , float(train_acc), end='\t')

            self.train_acc_metric.reset_states()

            # VALIDATION LOOP with BATCH
            for x_batch_val, y_batch_val in self.val_dataset:
                val_logits = self.l_step_model(x_batch_val, training=False)
                self.val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            tf.print("Validation acc: " , float(val_acc), end='\t')

            tf.print("Time taken: " , (time.time() - start_time))

    def test(self):
        predict = self.l_step_model(self.test_x)

        print(predict)

        print("======= report ==========")
        print("MAE: ",mean_absolute_error(y_true=self.test_y, y_pred=predict))
        print("=========================")
        print("ME: ", np.mean(np.subtract(self.test_y, predict)))
        print("std: ", np.std(np.subtract(self.test_y, predict)))
        print("=========================")
        print("relative error: ", np.mean(np.divide(np.absolute(np.subtract(self.test_y, predict)), self.test_y))*100)
        print("=========================")        
        print("r2 score: ", r2_score(y_true=self.test_y, y_pred=predict))
        print("=========================")

if __name__ == '__main__':
    setup_gpu()
    model = StepModel()
    model.train()
    model.test()
