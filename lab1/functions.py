import numpy as np


def rosenbrock(x):

    x = np.asarray(x)

    return np.sum(
        100 * (x[1:] - x[:-1] ** 2) ** 2
        + (1 - x[:-1]) ** 2
    )