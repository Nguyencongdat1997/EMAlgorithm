import numpy as np
from algorithms import *


def create_data():
    """
        Create dataset with 30 draws and 20 flips in each draws
    :return: dataset, (theta_A, theta_B)
    """
    theta_A, theta_B = np.random.rand(2)
    pi = np.random.uniform(0, 1)
    X = []

    for i in range(30):
        coin_choice = np.random.rand()
        theta = theta_A if coin_choice < pi else theta_B
        X.append(np.random.choice([0, 1], 20, p=[theta, 1-theta]))

    return np.array(X), (theta_A, theta_B)


def error_score(predict_thetas, ground_truth_thetas):
    """
        Calculate the difference between predicted thetas and their ground truth values
    :param predict_thetas: predicted thetas. Type: tuple of (theta_A, theta_B)
    :param ground_truth_thetas: ground truth thetas. Type: tuple of (theta_A, theta_B)
    :return: difference. Type: float
    """
    return min(abs(predict_thetas[0] - ground_truth_thetas[0])+abs(predict_thetas[1]-ground_truth_thetas[1]),
               abs(predict_thetas[1] - ground_truth_thetas[0])+abs(predict_thetas[0]-ground_truth_thetas[1]))

def evaluate(param, num_try=10):
    """
        Run multiple trials to get the average score of using given set of param.
    :param param: Given parameters. Type: a dict, includes {'init_func', 'converge_threshold', 'converge_steps}
    :param num_try: Number of trials
    :return: The average score of error.
    """
    total_error_score = 0
    for i in range(num_try):
        samples, ground_truth_theta = create_data()
        predict_theta = multi_em_func(samples,
                                      init_func=param['init_func'],
                                      converge_threshold=param['converge_threshold'],
                                      converge_steps=param['converge_steps'])
        total_error_score += error_score(predict_theta, ground_truth_theta)
    return total_error_score/num_try

if __name__ == '__main__':
    print('Initializing params...')
    init_func_list = [random_init, diverge_init, equal_init, greedy_init]
    converge_threshold_list = [1e-5, 1e-10]
    converge_steps_list = [10, 15, 20]
    param_list = []
    for x in init_func_list:
        for y in converge_threshold_list:
            for z in converge_steps_list:
                param_list.append({'init_func': x, 'converge_threshold': y, 'converge_steps': z})

    print('Tuning params...')
    evaluations = [evaluate(param) for param in param_list]

    print('Tuning done...')
    print('Best sets of tuned params:')
    best_evaluations = [eval[0] for eval in sorted(enumerate(evaluations), key=lambda e: -e[1])[-3:]]
    for x in best_evaluations:
        print('Param', param_list[x], 'Error', evaluations[x])



