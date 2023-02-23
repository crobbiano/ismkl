import numpy as np
import scipy.spatial.distance as sd

class KernelMatrix:
    '''
    This is an object that contains the multi-kernel matrix as defined in the paper.
    It takes in a set of kernel names and parameters and produces an NxN*K (short and fat) matrix that is all the kernel
    matrices concatenated together. There may be a better data structure for this.
    '''
    def __init__(self, X: np.array, Y: np.array, kernels: dict, matrix: np.array = None):
        self.X = X
        self.Y = Y
        # kernels needs to look like a dict:
        #   {'<kernel_name>': {'param1': <value>, 'param2': <value>, etc}
        self.kernels = kernels
        # can prepopulate the matrix
        self._matrix = matrix

        self.supported_kernels = ['quartic', 'gaussian', 'polynomial', 'linear', 'tanh']

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self._generate_kernel_matrices()
        return self._matrix

    def _generate_kernel_matrices(self):
        dist_mat = sd.cdist(self.X.T, self.Y.T)
        matrices = []
        for k, v in self.kernels.items():
            match k:
                case 'quartic':
                    tmp = np.square((1-np.square(dist_mat)/(2*v['param1']**2)))
                    tmp[np.square(dist_mat) >= 2*v['param1']**2] = 0
                case 'gaussian':
                    tmp = np.exp(-np.square(dist_mat)/(2*v['param1']**2))
                case 'polynomial':
                    tmp = (np.dot(self.X.T, self.Y) + v['param1'])**v['param2']
                case 'linear':
                    tmp = (np.dot(self.X.T, self.Y) + v['param1'])
                case 'tanh':
                    tmp = np.tanh(v['param1'] + v['param2'] * np.dot(self.X.T, self.Y))
                case _:
                    raise NotImplementedError(f'Kernel type {k} does not match any of the supported kernel types:'
                                              f' {self.supported_kernels}')

            matrices.append(tmp)

        return np.concatenate(matrices, axis=1)


if __name__ == '__main__':
    X = np.random.random((1000, 50))
    Y = np.random.random((1000, 28))
    kernels = {'gaussian': {'param1': 1},
               'linear': {'param1': 1},
               'polynomial': {'param1': 1, 'param2': 1},
               'tanh': {'param1': 1, 'param2': 1},
               'quartic': {'param1': 1},
               }
    km = KernelMatrix(X, Y, kernels)
    print(km.matrix.shape)
    print(km.matrix)