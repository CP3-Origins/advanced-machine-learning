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
        self.g = 0
        self.v0 = 0

    def call(self, input_shape):
        t = input_shape
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