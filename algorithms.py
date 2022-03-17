import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import warnings


def em_func(X, init_thetas=None, converge_threshold=1e-8, converge_steps=5, max_steps=1000):
    """
        EM algorithm
    :param X: sample data. Type: A tensor with shape (number_of_draws, number_of_flip_in_each_draw)
    :param init_thetas: Initialization of theta_A and theta_B. Type: a tuple of (init_theta_A, init_theta_B)
    :param converge_threshold: The threshold to decide if there is param updates in each EM iteration or not. Type: float
    :param converge_steps: Number of steps without param updates to try before terminate the process. Type: int
    :param max_steps: Maximum number of steps to run. Type: int
    :return: A tuple of estimated (theta_A, theta_B)
    """
    N, M = X.shape # N: number of draws, M: number of flips in each draw

    theta_A, theta_B = init_thetas if init_thetas else sample_init()[0]
    non_changed_steps = 0
    steps = 0
    while True:
        steps += 1

        # E-Step
        head_counts = np.sum(X, axis=1)
        tail_counts = M - head_counts
        prob_X_given_A = np.power(theta_A, head_counts) * np.power(1 - theta_A, tail_counts)
        prob_X_given_B = np.power(theta_B, head_counts) * np.power(1 - theta_B, tail_counts)

        gamma_A = prob_X_given_A / (prob_X_given_A + prob_X_given_B)
        gamma_A[np.isnan(gamma_A)] = 0.5 # in case of zero probability
        gamma_B = 1 - gamma_A

        weighted_head_A = np.sum(gamma_A * head_counts) + 1e-10 # +1e-10 for smoothing
        weighted_tail_A = np.sum(gamma_A * tail_counts) + 1e-10
        weighted_head_B = np.sum(gamma_B * head_counts) + 1e-10
        weighted_tail_B = np.sum(gamma_B * tail_counts) + 1e-10

        # M-step
        next_theta_A = weighted_head_A / (weighted_head_A + weighted_tail_A)
        next_theta_B = weighted_head_B / (weighted_head_B + weighted_tail_B)

        # Converge conditions
        if abs(next_theta_A - theta_A) + abs(next_theta_B - theta_B) < converge_threshold:
            theta_A, theta_B = next_theta_A, next_theta_B
            non_changed_steps += 1
            if non_changed_steps >= converge_steps or steps >= max_steps:
                break
        else:
            non_changed_steps = 0
            theta_A, theta_B = next_theta_A, next_theta_B

    return sorted([theta_A, theta_B], reverse=True)


def multi_em_func(X, init_func, converge_threshold=1e-8, converge_steps=5, max_steps=1000):
    """
        Running EM algorithm multiple times with different sets of initialization.
    :param X: sample data. Type: A tensor with shape (number_of_draws, number_of_flip_in_each_draw)
    :param init_func: Function from which sets of param are initialized.
    :param converge_threshold: The threshold to decide if there is param updates in each EM iteration or not. Type: float
    :param converge_steps: Number of steps without param updates to try before terminate the process. Type: int
    :param max_steps: Maximum number of steps to run. Type: int
    :return: The best estimated tuple of (theta_A, theta_B).
    """
    init_thetas_set = init_func(X)
    predicted_thetas_set = []

    # Get predicted thetas
    for init_theta in init_thetas_set:
        predicted_thetas_set.append(em_func(X,
                                            init_thetas=init_theta,
                                            converge_threshold=converge_threshold,
                                            converge_steps=converge_steps,
                                            max_steps=max_steps)
                                    )

    # Cluster thetas into groups, and pick the thetas that is nearest to the most common centroid
    predicted_thetas_set = np.array(predicted_thetas_set)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=min(3,len(predicted_thetas_set)), random_state=0, init='k-means++').fit(predicted_thetas_set)
        most_common_cluster_idx = max(set(kmeans.labels_.tolist()), key = kmeans.labels_.tolist().count)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, predicted_thetas_set)
    return predicted_thetas_set[closest[most_common_cluster_idx]]


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
    num_sample = min(8, samples.shape[0])
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