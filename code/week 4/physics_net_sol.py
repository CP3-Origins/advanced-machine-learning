import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import ops
from keras.initializers import Ones
from keras.models import Model
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Global constants
g = 9.7
v_0 = 20
scale = 5


def generate_data():
    """
    We generate data of a projectile motion in 1D
    """
    x_list = []
    y_list = []
    for i in range(10000):
        x = np.random.rand(100) * scale
        x_t = lambda t: v_0 * t - 1/2 * g * t**2
        x_list.append(x)
        y_list.append(x_t(x))

    return np.array(x_list), np.array(y_list)


class PhysicsNet(Model):
    # TODO: Make network with two subnets that learn to be g and v0
    def __init__(self):
        super().__init__()
        self.dense1a = Dense(units=100, activation='relu')
        self.dense1b = Dense(units=50, activation='relu')
        self.dense1c = Dense(units=10, activation='relu')
        self.dense1d = Dense(units=5, activation='relu')
        self.dense1e = Dense(units=1, activation='relu', kernel_initializer=Ones())
        self.dense2a = Dense(units=100, activation='relu')
        self.dense2b = Dense(units=50, activation='relu')
        self.dense2c = Dense(units=10, activation='relu')
        self.dense2d = Dense(units=5, activation='relu')
        self.dense2e = Dense(units=1, activation='relu', kernel_initializer=Ones())
        self.g = 0
        self.v0 = 0

    def call(self, input_shape):
        t = input_shape
        g = self.dense1a(t)
        g = self.dense1b(g)
        g = self.dense1c(g)
        g = self.dense1d(g)
        self.g = self.dense2e(g)
        v0 = self.dense2a(t)
        v0 = self.dense2b(v0)
        v0 = self.dense2c(v0)
        v0 = self.dense2d(v0)
        self.v0 = self.dense2e(v0)
        x = tf.math.multiply(self.v0, t) - tf.math.multiply(tf.math.multiply(tf.constant(0.5), self.g), tf.math.pow(t, 2))
        return x


x, y = generate_data()

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = PhysicsNet()

# Compile the model with categorical crossentropy loss for multi-class classification
model.compile(optimizer='adam', loss="mse", metrics=['mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

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

# Visualize accuracy
x = np.array([np.linspace(0, 5, 100)])
x_t = lambda t: v_0 * t - 1/2 * g * t**2
y = x_t(x[0])
plt.plot(x[0], model(x).numpy()[0])
plt.plot(x[0], model(x).numpy()[0])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()