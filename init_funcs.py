import numpy as np


def equal_init(samples):
    return [(0.5, 0.5)]


def diverge_init(samples):
    return [(0.9, 0.1)]


def sample_init(samples):
    return [(0.6, 0.5)]


def random_init(samples):
    num_init = 10
    return np.random.uniform(1e-9, 1.0, size=(num_init,2)).tolist()

def greedy_init(samples):
    num_sample = min(10, samples.shape[0])
    X = samples[:num_sample]
    head_counts = np.sum(X, axis=1)
    tail_counts = X.shape[1] - head_counts
    thetas = []

    for i in range(2**num_sample):
        binary_i = ('{0:0'+str(num_sample)+'b}').format(i)
        coin_choice = np.array([int(x) for x in binary_i])

        head_counts_A = np.sum(coin_choice * head_counts) + 1 # add-one smoothing
        tail_counts_A = np.sum(coin_choice * tail_counts) + 1
        head_counts_B = np.sum((1-coin_choice) * head_counts) + 1
        tail_counts_B = np.sum((1-coin_choice) * tail_counts) + 1

        theta_A = head_counts_A / (head_counts_A + tail_counts_A)
        theta_B = head_counts_B / (head_counts_B + tail_counts_B)

        thetas.append((theta_A, theta_B))

    return thetas



