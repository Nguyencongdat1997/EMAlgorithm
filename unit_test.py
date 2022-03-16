import numpy as np
from em import em_func

samples = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

print(em_func(samples))