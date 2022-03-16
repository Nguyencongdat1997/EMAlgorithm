import numpy as np
import unittest
import sys

from em import em_func


class TestEM(unittest.TestCase):
    def setUp(self):
        self.samples = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                                 [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                 [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

    def tearDown(self):
        pass

    def test_one_step_udpate(self):
        def sample_init():
            return 0.6, 0.5
        pred = em_func(self.samples, init_func=sample_init, converge_threshold=float('inf'))
        self.assertGreaterEqual(5e-3, abs(pred[0]-0.71)+abs(pred[1]-0.58))

    def test_converage(self):
        def sample_init():
            return 0.6, 0.5
        pred = em_func(self.samples, init_func=sample_init, converge_threshold=1e-9)
        self.assertGreaterEqual(5e-3, abs(pred[0]-0.80)+abs(pred[1]-0.52))


if __name__ == '__main__':
    unittest.main()