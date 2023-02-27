import unittest

from schema import SchemaMissingKeyError

from src.kernel_matrix import KernelMatrix
import numpy as np
from parameterized import parameterized


class KernelMatrixTest(unittest.TestCase):
    @parameterized.expand([
        ['gaussian', [{'gaussian': {'param1': 1}}]],
        ['linear', [{'linear': {'param1': 1}}]],
        ['polynomial', [{'polynomial': {'param1': 1, 'param2': 1}}]],
        ['tanh', [{'tanh': {'param1': 1, 'param2': 1}}]],
        ['quartic', [{'quartic': {'param1': 1}}]],
    ])
    def test_all_supported_work(self, kernel_name, kernel_dict):
        # Arrange
        num_x, num_y = 50, 28
        X = np.random.random((1000, num_x))
        Y = np.random.random((1000, num_y))
        km = KernelMatrix(X, Y, kernel_dict)
        # Act
        matrix = km.matrix
        # Assert
        self.assertEqual(matrix.shape[0], num_x)
        self.assertEqual(matrix.shape[1], num_y)
        self.assertTrue(kernel_name in km.supported_kernels)

    def test_unsupported_fails(self):
        # Arrange
        num_x, num_y = 50, 28
        x = np.random.random((1000, num_x))
        y = np.random.random((1000, num_y))
        kernel_dict = [{'not_a_kernel_name': {'param1': 1}}]
        # Act (and assert due to checking for raised error)
        self.assertRaises(SchemaMissingKeyError, KernelMatrix, **{'x': x, 'y': y, 'kernels': kernel_dict})
