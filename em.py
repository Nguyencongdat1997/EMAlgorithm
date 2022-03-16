import numpy as np


def equal_init():
    return 0.5, 0.5

def diverge_init():
    return 0.9, 0.1

def sample_init():
    return 0.6, 0.5


def em_func(X, init_func=sample_init, converge_threshold=1e-8):
    N, M = X.shape # N = 30, M = 10

    theta_A, theta_B = init_func()
    while True:
        # E-Step
        head_counts = np.sum(X, axis=1)
        tail_counts = M - head_counts
        prob_X_given_A = np.power(theta_A, head_counts) * np.power(1 - theta_A, tail_counts)
        prob_X_given_B = np.power(theta_B, head_counts) * np.power(1 - theta_B, tail_counts)
        gamma_A = prob_X_given_A / (prob_X_given_A + prob_X_given_B)
        gamma_B = 1 - gamma_A

        weighted_head_A = np.sum(gamma_A * head_counts)
        weighted_tail_A = np.sum(gamma_A * tail_counts)
        weighted_head_B = np.sum(gamma_B * head_counts)
        weighted_tail_B = np.sum(gamma_B * tail_counts)

        # M-step
        next_theta_A = weighted_head_A / (weighted_head_A + weighted_tail_A)
        next_theta_B = weighted_head_B / (weighted_head_B + weighted_tail_B)

        # Converge conditions
        if abs(next_theta_A - theta_A) + abs(next_theta_B - theta_B) < converge_threshold:
            theta_A, theta_B = next_theta_A, next_theta_B
            break
        else:
            theta_A, theta_B = next_theta_A, next_theta_B
    return theta_A, theta_B

