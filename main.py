import numpy as np
from algorithms import *
from network_services import APIConnector


if __name__ == '__main__':
    api_connector = APIConnector()

    print('Collecting draws...')
    dataset = []
    for i in range(30):
        dataset.append(api_connector.get_one_draw())
    dataset = np.array(dataset)

    print('Estimating thetas...')
    thetas = multi_em_func(dataset, init_func=greedy_init, converge_threshold=1e-10, converge_steps=20)

    print('Results:', thetas)