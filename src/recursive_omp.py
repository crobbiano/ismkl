import numpy as np
from tqdm import tqdm


class RecursiveOMP:
    '''
    Implements recursive orthogonal matching pursuit from an Azimi paper
    '''

    def __init__(self, d, x, y, residual_norm, coeff_tolerance=1e-4, max_iterations: int = 1000):
        self.d = d
        self.x = x
        self.y = y
        self.residual_norm = residual_norm
        self.coeff_tolerance = coeff_tolerance
        self.max_iterations = max_iterations

    def run(self):
        if len(self.x) == 0:
            self.x = np.zeros((self.d.shape[1], self.y.shape[1]))
        elif self.x.shape[0] < self.d.shape[1]:
            self.x = np.concatenate([self.x, np.zeros((self.d.shape[1] - self.x.shape[0], self.x.shape[1]))], axis=0)

        for i in tqdm(range(self.y.shape[1])):
            y_l = np.expand_dims(self.y[:, i], axis=-1)
            # y_l = self.y[ i, :]
            residual_norm_l = self.residual_norm * np.linalg.norm(y_l)
            # indices = np.squeeze(np.argwhere(self.x[i, :] == 0))
            indices = np.squeeze(np.argwhere(self.x[:, i] != 0))
            # indices = np.argwhere(self.x[:, i] == 0)

            if len(indices) == 0:
                # init vars
                r = y_l
                dt = np.expand_dims(np.asarray([]), axis=-1)
                # find best match
                idx = np.expand_dims(np.argmax(np.abs(np.dot(r.T, self.d))), axis=-1)
                dt = np.concatenate([dt, self.d[:, idx]], axis=0)
                # dt = np.concatenate([dt, self.d[idx, :]], axis=1)
                indices = np.concatenate([indices, idx], axis=0)

                # Update the filter
                q = np.divide(dt.T, np.dot(dt.T, dt))
                alpha = np.dot(q, y_l)

                # Update coefficients
                x_l = alpha
                r = r - alpha * dt
            else:
                # build dictionary and filter
                dt = self.d[:, indices]
                # dt = self.d[ indices, :]
                q = np.dot(np.linalg.inv(np.dot(dt.T, dt)), dt.T)

                # compute coefficients and residual
                x_l = np.dot(q, y_l)
                r = y_l - np.dot(dt, x_l)

            # add atoms to d until norm of residual falls below thresholds
            xbest = x_l
            indices_best = []
            num_iteration = 0
            rprev = r
            while (np.linalg.norm(r) > residual_norm_l) and (num_iteration < self.max_iterations):
                num_iteration += 1
                idx = np.expand_dims(np.argmax(np.abs(np.dot(r.T, self.d))), axis=-1)
                d_l = self.d[:, idx]
                indices = np.concatenate([indices, idx], axis=0)

                # update filters
                b = np.dot(q, d_l)
                d_tilde = d_l - np.dot(dt, b)
                q_l = d_tilde/np.linalg.norm(d_tilde)**2
                alpha = np.dot(q_l.T, y_l)
                dt = np.concatenate([dt, d_l], axis=1)
                q = np.concatenate([q - np.dot(b, q_l.T), q_l.T], axis=0)

                # update coefficients
                x_l = np.concatenate([x_l-alpha*b, alpha], axis=0)
                r = r-alpha*d_tilde
                if np.linalg.norm(rprev) > np.linalg.norm(r):
                    xbest = x_l
                    indices_best = indices
                rprev = r

            if len(indices_best) == 0:
                indices_best = indices
                xbest = x_l

            # fill in the coefficients
            self.x[indices_best, i] = xbest.flatten()

        self.x[np.abs(self.x) < self.coeff_tolerance] = 0

        return self.x

if __name__ == "__main__":
    from src.kernel_matrix import KernelMatrix
    num_x, num_y, N = 50, 28, 1000
    features = np.random.random((N, num_x))
    kernel_dict = {
        'gaussian': {'param1': 1},
        'quartic': {'param1': 1},
    }
    values = np.random.randint(0, 4, num_x)
    n_values = np.max(values) + 1
    L0 = np.eye(n_values)[values]
    K00 = KernelMatrix(features, features, kernel_dict).matrix
    # Act
    W0 = RecursiveOMP(K00, [], L0, residual_norm=0.1).run()
    # Assert
