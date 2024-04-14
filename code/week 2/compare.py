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


if __name__ == "__main__":
    main()

