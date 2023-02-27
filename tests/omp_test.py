import unittest
import numpy as np
from src.kernel_matrix import KernelMatrix
from src.recursive_omp import RecursiveOMP


class RecursiveOMPTest(unittest.TestCase):

    def test_omp_runs(self):
        # Arrange
        num_x, num_y, N = 50, 28, 1000
        features = np.random.random((N, num_x))
        kernel_dict = [
            {'gaussian': {'param1': 1}},
            {'quartic': {'param1': 1}},
        ]
        values = np.random.randint(0, 4, num_x)
        n_values = np.max(values) + 1
        L0 = np.eye(n_values)[values]
        K00 = KernelMatrix(features, features, kernel_dict).matrix
        # Act
        W0 = RecursiveOMP(K00, [], L0, residual_norm=0.1).run()
        # Assert
        self.assertEqual(W0.shape, (100, 4))


    def test_validate_omp_is_correct(self):
        features = np.asarray([[1, 2, 3, 4],
                               [2, 3, 4, 5],
                               [3, 4, 5, 6],
                               [4, 5, 6, 7],
                               [5, 6, 7, 8],
                               [6, 7, 8, 9],
                               [7, 8, 9, 10],
                               ])
        kernel_dict = [
            {'gaussian': {'param1': .5}},
            {'quartic': {'param1': 1}},
        ]
        values = np.asarray([0, 1, 0, 2])
        n_values = np.max(values) + 1
        L0 = np.eye(n_values)[values]
        K00 = KernelMatrix(features, features, kernel_dict).matrix
        # Act
        W0 = RecursiveOMP(K00, [], L0, residual_norm=0.1).run()
        expected = np.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
        self.assertTrue(np.all(W0 - expected < np.finfo(float).eps))

