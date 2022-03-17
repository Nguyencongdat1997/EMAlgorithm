import numpy as np
import unittest
import sys

from algorithms import *


class TestEM(unittest.TestCase):
    def setUp(self):
        self.samples = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                                 [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                 [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

    def tearDown(self):
        pass

    def test_one_step_update(self):
        pred = em_func(self.samples, init_thetas=(0.6, 0.5), converge_threshold=float('inf'), converge_steps=1)
        self.assertGreaterEqual(5e-3, abs(pred[0]-0.71)+abs(pred[1]-0.58))

    def test_converge(self):
        pred = em_func(self.samples, init_thetas=(0.6, 0.5), converge_threshold=1e-9, converge_steps=10)
        self.assertGreaterEqual(5e-3, abs(pred[0]-0.80)+abs(pred[1]-0.52))

    def test_boundary_value(self):
        error = False
        try:
            pred = em_func(self.samples, init_thetas=(0, 1), converge_threshold=1e-9, converge_steps=10)
        except Exception:
            error = True
        self.assertEqual(error, False)

    def test_equal_init(self):
        init_thetas_set = equal_init(self.samples)
        error = False
        try:
            for init_thetas in init_thetas_set:
                pred = em_func(self.samples, init_thetas=init_thetas)
        except Exception:
            error = True
        self.assertEqual(error, False)

    def test_diverge_init(self):
        init_thetas_set = diverge_init(self.samples)
        error = False
        try:
            for init_thetas in init_thetas_set:
                pred = em_func(self.samples, init_thetas=init_thetas)
        except Exception:
            error = True
        self.assertEqual(error, False)

    def test_sample_init(self):
        init_thetas_set = sample_init(self.samples)
        error = False
        try:
            for init_thetas in init_thetas_set:
                pred = em_func(self.samples, init_thetas=init_thetas)
        except Exception:
            error = True
        self.assertEqual(error, False)

    def test_random_init(self):
        init_thetas_set = random_init(self.samples)
        error = False
        try:
            for init_thetas in init_thetas_set:
                pred = em_func(self.samples, init_thetas=init_thetas)
        except Exception:
            error = True
        self.assertEqual(error, False)

    def test_greedy_init(self):
        init_thetas_set = greedy_init(self.samples)
        error = False
        try:
            for init_thetas in init_thetas_set:
                pred = em_func(self.samples, init_thetas=init_thetas)
        except Exception:
            error = True
        self.assertEqual(error, False)


if __name__ == '__main__':
    unittest.main()