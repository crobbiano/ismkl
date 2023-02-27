import logging

import numpy as np
import scipy.spatial.distance as sd
from schema import Schema, And, Or


class KernelMatrix:
    '''
    This is an object that contains the multi-kernel matrix as defined in the paper.
    It takes in a set of kernel names and parameters and produces an NxN*K (short and fat) matrix that is all the kernel
    matrices concatenated together. There may be a better data structure for this.
    '''

    def __init__(self, x: np.array, y: np.array, kernels: dict, matrix: np.array = None):
        self.x = x
        self.y = y
        # kernels needs to look like a dict:
        #   {'<kernel_name>': {'param1': <value>, 'param2': <value>, etc}
        self.kernels = kernels
        # can prepopulate the matrix
        self._matrix = matrix

        self.supported_kernels = ['quartic', 'gaussian', 'polynomial', 'linear', 'tanh']

        self._validate_kernels()

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self._generate_kernel_matrices()
        return self._matrix

    def _validate_kernels(self):
        kernels_schema = Schema({And(str, lambda k: k in self.supported_kernels): {And(str, lambda m: 'param' in m): Or(int, float)}})
        for kern in self.kernels:
            kernels_schema.validate(kern)

    def _generate_kernel_matrices(self):
        dist_mat = sd.cdist(self.x.T, self.y.T)
        matrices = []
        for kern in self.kernels:
            for k, v in kern.items():
                match k:
                    case 'quartic':
                        tmp = np.square((1 - np.square(dist_mat) / (2 * v['param1'] ** 2)))
                        tmp[np.square(dist_mat) >= 2 * v['param1'] ** 2] = 0
                    case 'gaussian':
                        tmp = np.exp(-np.square(dist_mat) / (2 * v['param1'] ** 2))
                    case 'polynomial':
                        tmp = (np.dot(self.x.T, self.y) + v['param1']) ** v['param2']
                    case 'linear':
                        tmp = (np.dot(self.x.T, self.y) + v['param1'])
                    case 'tanh':
                        tmp = np.tanh(v['param1'] + v['param2'] * np.dot(self.x.T, self.y))

                matrices.append(tmp)

        # return np.concatenate(matrices, axis=1)
        return np.dstack(matrices).reshape((tmp.shape[0],-1))
