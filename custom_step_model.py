from distutils.log import fatal
from msilib.schema import Directory
from subprocess import call
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
import keras_tuner
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

class StepHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = Sequential()
        hp_filters = hp.Choice('filters', [ 64, 128, 256, 512])
        hp_kernel = hp.Choice('kernel', [2, 3, 4, 5, 6, 7, 8])
        model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, padding='same', activation='swish', input_shape=[612,1]))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, padding='same', activation='swish'))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, padding='same', activation='swish'))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, padding='same', activation='swish'))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        hp_units = hp.Choice('units', [1024])
        model.add(Dense(units=hp_units, activation='swish'))
        model.add(BN())
        model.add(Dense(units=hp_units, activation='swish'))
        model.add(BN())
        model.add(Dense(units=hp_units, activation='swish'))
        model.add(BN())
        model.add(Dense(units=hp_units, activation='swish'))
        model.add(BN())
        model.add(Dense(1, activation=None))
        model.summary()

        return model

    def fit(self, hp, model, x, y, validation_data,callbacks=None, **kwargs):

        #batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        batch_size = hp.Int("batch_size", 32, 64, step=1, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
        
        # Define the optimizer.
        #optimizer = Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3))
        optimizer = Adam(hp.Float("learning_rate", 1e-4, 1e-3, sampling="log", default=1e-3))
        loss_fn = MSE()
        epoch_loss_metric = MSE_metrics()

        def run_train_step(x_data, y_data):
            with tf.GradientTape() as tape:
                logits = model(x_data)
                loss = loss_fn(y_data, logits)
                # Add any regularization losses.

                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        def run_val_step(x_data, y_data):
            logits = model(x_data)
            #loss = loss_fn(y_data, logits)
            # Update the metric.
            epoch_loss_metric.update_state(y_data, logits)

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model

        # Record the best validation loss value
        best_epoch_loss = float("inf")

        # The custom training loop.
        for epoch in range(2):
            print(f"Epoch: {epoch}")

            for x_data, y_data in train_ds:
                run_train_step(x_data, y_data)

            # Iterate the validation data to run the validation step.
            for x_data, y_data in validation_data:
                run_val_step(x_data, y_data)

            # Calling the callbacks after epoch.
            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"mse_metric": epoch_loss})
            epoch_loss_metric.reset_states()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        # Return the evaluation metric value.
        return best_epoch_loss

class StepModel():
    def __init__(self) -> None:
        self.epochs = 2000
        self.learning_rate = 1.46e-4
        self.batch_size = 64
        self.input_size = None

        self.train_dataset =None
        self.val_dataset =None
        self.test_dataset =None   
        self.load_dataset()

        self.step_optimizer = None
        self.step_loss = None
        self.setup_optimizers()

        self.train_acc_metric = None 
        self.val_acc_metric = None
        self.setup_metrics()

        self.step_model = None
        self.tuner = None

        self.step_model = self.build_cnn_model()
        
    def tune_model(self):
        self.l_step_model = StepHyperModel()
        self.tuner = keras_tuner.Hyperband(
            objective=keras_tuner.Objective("mse_metric", "min"),
            hypermodel=self.l_step_model,
            hyperband_iterations=5,
            max_epochs=2000,
            factor=3,
            directory= 'results',
            overwrite=True
        )

        self.tuner.search(x=self.train_x, y=self.train_y, validation_data=(self.val_x, self.val_y))

        best_hps = self.tuner.get_best_hyperparameters()[0]
        print(best_hps.values)

        best_model = self.tuner.get_best_models()[0]
        best_model.summary()

    def scale_data(self, train_data, val_data, test_data):
        print("==== scaling DATA ====")
        
        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
        scaler = RobustScaler()

        total_data = np.concatenate((train_data, val_data), axis=0)

        scaler.fit(total_data)

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
        y_data = data_list[:,0]

        train_x, test_x, train_y, test_y =  train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        train_x, val_x, train_y, val_y =  train_test_split(train_x, train_y, test_size=0.2, random_state=42)

        train_x, val_x, test_x = self.scale_data(train_x, val_x, test_x)
        self.input_size = len(train_x[0])

        train_x = train_x.reshape(-1, self.input_size, 1)
        val_x = val_x.reshape(-1, self.input_size, 1)
        test_x = test_x.reshape(-1, self.input_size, 1)

        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
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
        model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='swish', input_shape=[self.input_size,1]))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='swish'))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='swish'))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='swish'))
        model.add(BN())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=1024, activation='swish'))
        model.add(BN())
        model.add(Dense(units=1024, activation='swish'))
        model.add(BN())
        model.add(Dense(units=1024, activation='swish'))
        model.add(BN())
        model.add(Dense(units=1024, activation='swish'))
        model.add(BN())
        model.add(Dense(1, activation=None))
        model.summary()

        return model

    def setup_optimizers(self):
        print("==== setting up OPT ====")
        self.step_optimizer = Adam(learning_rate=self.learning_rate)
        self.step_loss = MSE()

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
                    logits = self.step_model(x_batch_train, training=True)
                    step_loss_value = self.step_loss(y_batch_train, logits)

                grads = tape.gradient(step_loss_value, self.step_model.trainable_variables)

                self.step_optimizer.apply_gradients(zip(grads, self.step_model.trainable_variables))

            print("train loss : %.4f" % float(step_loss_value), end='\t')

            # VALIDATION LOOP with BATCH
            for x_batch_val, y_batch_val in self.val_dataset:
                val_logits = self.step_model(x_batch_val, training=False)
                self.val_acc_metric.update_state(y_batch_val, val_logits)

            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            
            print("val loss: %.4f" % (float(val_acc),), end='\t')

            self.train_dataset = self.org_train_dataset.shuffle(buffer_size=128, reshuffle_each_iteration=True).batch(self.batch_size)

            print("Time taken: %.2fs" % (time.time() - epoch_time))

        print("Whole Time taken: %.2fs" % (time.time() - start_time))

    def test(self):
        predict = self.step_model(self.test_x)
    
        predict = predict.numpy()

        for i in range(len(self.test_y)):
            print("real : ", self.test_y[i], " predict : " , predict[i])

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
    model.tune_model()
    #model.train()
    #model.test()
