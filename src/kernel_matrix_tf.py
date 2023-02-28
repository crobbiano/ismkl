import logging

import numpy as np
import scipy.spatial.distance as sd
from schema import Schema, And, Or
import tensorflow as tf
import tensorflow_probability as tfp


class KernelMatrix:
    '''
    This is an object that contains the multi-kernel matrix as defined in the paper.
    It takes in a set of kernel names and parameters and produces an NxN*K (short and fat) matrix that is all the kernel
    matrices concatenated together. There may be a better data structure for this.
    '''

    def __init__(self, x: np.array, y: np.array, kernels: dict, matrix: np.array = None):
        self.X = x.T
        self.Y = y.T
        # kernels needs to look like a dict:
        #   {'<kernel_name>': {'param1': <value>, 'param2': <value>, etc}
        self.kernels = kernels
        # can prepopulate the matrix
        self._matrix = matrix

        self.supported_kernels = ['quartic', 'gaussian', 'polynomial', 'linear', 'tanh']

        self._validate_kernels()
        # raise NotImplementedError(f'Kernel type {k} does not match any of the supported kernel types:'
        #                   f' {self.supported_kernels}')

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self._generate_kernel_matrices()
        return self._matrix

    def _validate_kernels(self):
        kernels_schema = Schema({And(str, lambda k: k in self.supported_kernels): {And(str, lambda m: 'param' in m): Or(int, float)}})
        for kern in self.kernels:
            kernels_schema.validate(kern)

    @tf.function
    def _generate_kernel_matrices(self):
        matrices = []
        for kern in self.kernels:
            for k, v in kern.items():
                match k:
                    # case 'quartic':
                    #     tmp = np.square((1 - np.square(dist_mat) / (2 * v['param1'] ** 2)))
                    #     tmp[np.square(dist_mat) >= 2 * v['param1'] ** 2] = 0
                    case 'gaussian':
                        kfunc = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=v['param1'])
                    case 'polynomial':
                        kfunc = tfp.math.psd_kernels.Polynomial(shift=v['param1'], exponent=v['param2'])
                    case 'linear':
                        kfunc = tfp.math.psd_kernels.Linear(shift=v['param1'])
                    # case 'tanh':
                    #     tmp = np.tanh(v['param1'] + v['param2'] * np.dot(self.X.T, self.Y))

                matrices.append(tf.cast(kfunc.matrix(self.X, self.Y), tf.float32))

        ts = []
        poss = []
        for idx, matrix in enumerate(matrices):
            ts.append(tf.transpose(matrix, perm=[1, 0]))
            poss.append(tf.convert_to_tensor(list(range(idx, len(matrices)*ts[idx].shape[0], len(matrices))), dtype=tf.int32))
        interleaved = tf.dynamic_stitch(poss, ts)
        interleaved = tf.transpose(interleaved, perm=[1, 0])

        return interleaved


if __name__ == '__main__':
    num_samples = 100
    num_classes = 10
    features = np.random.random((1000, num_samples))
    labels = np.random.randint(0, num_classes - 1, num_samples)
    kernel_dicts = [
        {'gaussian': {'param1': 1}},
        {'linear': {'param1': 1}},
        {'polynomial': {'param1': 2, 'param2': 1}},
    ]
    L0 = np.eye(num_classes)[labels]
    K00 = KernelMatrix(features, features, kernel_dicts).matrix
    # Act
