import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


class NeuralSolver:
    """ A simple neural network to solve ODEs """
    def __init__(self, f0, func):
        self.f0 = f0
        self.func = func
        self.inf_s = np.sqrt(np.finfo(np.float32).eps)

        # Network Parameters
        # TODO: Build a suitable network and set the hyperparamters
        self.epochs = 100
        self.display_step = 10
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
        )
        self.model = Sequential(
            [
                Input(shape=(1,)),
                Dense(8, activation="relu", name="layer1"),
                Dense(32, activation="relu", name="layer2"),
                Dense(32, activation="relu", name="layer3"),
                Dense(8, activation="relu", name="layer4"),
                Dense(1, name="output"),
            ]
        )
        self.train()
        print(self.model.summary())

    def solution(self, x):
        """ Call this function to use the trained model"""
        tensor_input = tf.cast([x], tf.float32)
        return tensor_input*self.model(tensor_input) + self.f0

    def custom_loss(self):
        """ Custom loss function """
        summation = []
        for x in np.linspace(0, 2, 10):
            dNN = (self.solution(x + self.inf_s) - self.solution(x)) / self.inf_s
            summation.append((dNN - self.func(x)) ** 2)
        return tf.reduce_sum(tf.abs(summation))

    def train(self):
        """ Custom training method """
        def train_step():
            with tf.GradientTape() as tape:
                loss = self.custom_loss()
            trainable_variables = self.model.trainable_weights
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Train the model and write out logs
        for i in range(self.epochs):
            train_step()
            if i % self.display_step == 0:
                print(f"Epoch: {i}")
                print(f"loss: {self.custom_loss()} ")

    @classmethod
    def func1(cls):
        def f1(x):
            return 2 * x

        return cls(1, f1)

    @classmethod
    def func2(cls):
        def f2(x):
            return x ** 2

        return cls(1, f2)

    @classmethod
    def func3(cls):
        def f3(x):
            return x ** 2 - 2 * x

        return cls(1, f3)

def compare_result(model, true):
    """
    Compare with the approximate solution from the model
    and the analytical solution.

    Args:
    - model: Trained model
    - true: The true model
    """
    X = np.linspace(0, 2, 100)
    result = []
    for i in X:
        result.append(model(i).numpy()[0][0])
    S = true(X)
    plt.plot(X, S, label="Original Function")
    plt.plot(X, result, label="Neural Net Approximation")
    plt.ylim([-1, 6])
    plt.legend(loc=2, prop={'size': 10})
    plt.show()


def main():
    # Functions we consider
    def f1_sol(x):
        return x ** 2 + 1

    def f2_sol(x):
        return x ** 3 / 3 + 1

    def f3_sol(x):
        return x ** 3 / 3 - x ** 2 + 1

    # End of functions

    # Train and compare
    model_1 = NeuralSolver.func1().solution
    compare_result(model_1, f1_sol)
    model_2 = NeuralSolver.func2().solution
    compare_result(model_2,f2_sol)
    model_3 = NeuralSolver.func3().solution
    compare_result(model_3, f3_sol)


if __name__ == "__main__":
    main()
