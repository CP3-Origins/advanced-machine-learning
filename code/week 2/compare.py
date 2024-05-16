import matplotlib.pyplot as plt
import numpy as np

from reinforcement import NeuralSolver as RNS
from supervised import NeuralSolver as SNS


def main():
    # True solutions
    def f1_sol(x):
        return x ** 2 + 1

    def f2_sol(x):
        return x ** 3 / 3 + 1

    def f3_sol(x):
        return x ** 3 / 3 - x ** 2 + 1

    # TODO: Call the Neuralsolver classes to get the results and compare with a nice plot
    model_11 = RNS.func3().solution
    model_12 = SNS.func3().model

    X = np.linspace(0, 5, 100)
    result_12 = model_12(X)
    result_11 = []
    for i in X:
        result_11.append(model_11(i).numpy()[0][0])
    S = f3_sol(X)
    plt.plot(X, S, label="Original Function")
    plt.plot(X, result_11, label="Neural Net RL")
    plt.plot(X, result_12, label="Neural Net SL")
    plt.ylim([-1, 10])
    plt.legend(loc=2, prop={'size': 10})
    plt.show()


if __name__ == "__main__":
    main()

