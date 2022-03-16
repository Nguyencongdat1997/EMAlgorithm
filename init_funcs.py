import numpy as np


def equal_init():
    return 0.5, 0.5


def diverge_init():
    return 0.9, 0.1


def sample_init():
    return 0, 0.5


def random_init():
    return np.random.uniform(1e-9, 1.0, size=2)
