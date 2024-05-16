import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import L1


class Network:
    """ A simple neural network to imitate a wave function"""
    def __init__(self):
        # Size of each x and y input
        self.data_points = 200
        x, y = self.generate_data()

        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42
        )
        # Network Parameters
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
        )
        # TODO: Add regularizers to prevent over-fitting
        self.model = Sequential(
            [
                Input(shape=(self.data_points,)),
                Dense(128, activation="relu"),
                Dense(128, activation="relu"),
                Dense(128, activation="relu"),
                Dense(128, activation="relu"),
                Dense(128, activation="relu"),
                Dense(self.data_points, name="output"),
            ]
        )
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mse'])
        history = self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
        print(self.model.summary())

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.title('Model MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.show()

    def generate_data(self):
        """
        We generate data of a wavefunction norm of
        an infinite well with a width of l
        """
        x_list = []
        y_list = []
        for i in range(10000):
            l = 2
            n = 10
            x = np.random.rand(self.data_points, 1) * l
            psi = lambda x: np.sqrt(2 / l) * np.sin(x * n * np.pi / l)
            x_list.append(x)
            y_list.append(psi(x)**2)

        return np.array(x_list), np.array(y_list)


if __name__ == '__main__':
    Network()
