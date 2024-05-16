import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


class NeuralSolver:
    """ A simple neural network to solve ODEs """
    def __init__(self, sol):
        self.sol = sol
        x, y = self.generate_data()

        # TODO: Use x, y as input and then train a supervised model
        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42
        )
        # Network Parameters
        # TODO: Build a suitable network and set the hyperparamters
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
                Dense(32, activation="relu", name="layer1"),
                Dense(32, activation="relu", name="layer2"),
                Dense(32, activation="relu", name="layer3"),
                Dense(1, name="output"),
            ]
        )
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        print(self.model.summary())

    def generate_data(self):
        data_points = int(10e3)
        scale = 2
        x = np.random.rand(data_points, 1) * scale
        y = self.sol(x)
        return x, y

    @classmethod
    def func1(cls):
        def f1_sol(x):
            return x ** 2 + 1

        return cls(f1_sol)

    @classmethod
    def func2(cls):
        def f2_sol(x):
            return x ** 3 / 3 + 1

        return cls(f2_sol)

    @classmethod
    def func3(cls):
        def f3_sol(x):
            return x ** 3 / 3 - x ** 2 + 1

        return cls(f3_sol)



def compare_result(model, true):
    """
    Compare with the approximate solution from the model
    and the analytical solution.

    Args:
    - model: Trained model
    - true: The true model
    """
    X = np.linspace(0, 2, 100)
    result = model(X)
    S = true(X)
    plt.plot(X, S, label="Original Function")
    plt.plot(X, result, label="Neural Net Approximation")
    plt.ylim([-1, 6])
    plt.legend(loc=2, prop={'size': 10})
    plt.show()


def main():
    # Solutions to replicate
    def f1_sol(x):
        return x ** 2 + 1

    def f2_sol(x):
        return x ** 3 / 3 + 1

    def f3_sol(x):
        return x ** 3 / 3 - x ** 2 + 1

    # End of functions

    # Train and compare
    model_1 = NeuralSolver.func1().model
    compare_result(model_1, f1_sol)
    model_2 = NeuralSolver.func2().model
    compare_result(model_2,f2_sol)
    model_3 = NeuralSolver.func3().model
    compare_result(model_3, f3_sol)


if __name__ == "__main__":
    main()
