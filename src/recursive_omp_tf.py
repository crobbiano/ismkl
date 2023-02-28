import numpy as np
import tensorflow as tf
from tqdm import tqdm


class RecursiveOMP:
    '''
    Implements recursive orthogonal matching pursuit from an Azimi paper
    '''

    def __init__(self):
        pass


    # @tf.function
    def run(self, d, x, y, residual_norm,  coeff_tolerance=1e-4, max_iterations: int = 10):
        if len(x) == 0:
            x = tf.convert_to_tensor(tf.zeros((d.shape[1], y.shape[1])))
        elif x.shape[0] < d.shape[1]:
            x = tf.concat([x, tf.zeros((d.shape[1] - x.shape[0], x.shape[1]))], axis=0)

        for i in tqdm(range(y.shape[1])):
            y_l = tf.cast(tf.expand_dims(y[:, i], -1), tf.float32)
            # y_l = y[ i, :]
            residual_norm_l = residual_norm * tf.norm(y_l)
            # indices = tf.squeeze(tf.where(x[i, :] == 0))
            indices = tf.cast(tf.squeeze(tf.where(x[:, i] != 0)), tf.int32)

            if len(indices) == 0:
                # init vars
                r = y_l
                # find best match
                idx = tf.cast(tf.math.argmax(tf.abs(tf.tensordot(tf.transpose(r, [1, 0]), d, axes=1)), axis=1), tf.float32)
                dt = tf.gather(d, tf.cast(idx,tf.int32), axis=1)

                indices = tf.cast(idx, tf.int32)

                # Update the filter
                q = tf.divide(tf.transpose(dt, perm=[1, 0]), tf.tensordot(tf.transpose(dt, perm=[1, 0]), dt, axes=1))
                if len(q.get_shape()) == 1:
                    q = tf.expand_dims(q,  axis=0)
                alpha = tf.tensordot(q, y_l, axes=1)

                # Update coefficients
                x_l = alpha
                r = r - alpha * dt
            else:
                # build dictionary and filter
                dt = tf.gather(d, tf.cast(indices, tf.int32), axis=1)
                q = tf.tensordot(tf.linalg.inv(tf.tensordot(tf.transpose(dt, perm=[1, 0]), dt,axes=1)), tf.transpose(dt, perm=[1, 0]), axes=1)

                # compute coefficients and residual
                x_l = tf.tensordot(q, y_l, axes=1)
                r = y_l - tf.tensordot(dt, x_l, axes=1)

            xbest = x_l
            indices_best = tf.cast(tf.zeros(tf.cast(indices,tf.int32)), tf.int32)
            num_iteration = 0
            rprev = r
            while (tf.norm(r) > residual_norm_l) and (num_iteration < max_iterations):
                num_iteration += 1
                idx = tf.cast(tf.math.argmax(tf.abs(tf.tensordot(tf.transpose(r, [1, 0]), d, axes=1)), axis=1), tf.int32)
                d_l = tf.gather(d, idx, axis=1)

                # correct the shape of these soon to be vectors and matrices
                if len(d_l.get_shape()) == 1:
                    d_l = tf.expand_dims(d_l,  axis=-1)
                if len(indices) == 0:
                    indices = tf.expand_dims(indices, axis=-1)
                if len(idx.get_shape()) == 0:
                    idx = tf.expand_dims(idx, axis=-1)

                print(idx)
                indices = tf.concat([tf.cast(indices, tf.int32), tf.cast(idx, tf.int32)], axis=0)
                print(indices)

                # update filters
                b = tf.tensordot(q, d_l, axes=1)
                d_tilde = d_l - tf.tensordot(dt, b, axes=1)
                q_l = d_tilde/(tf.norm(d_tilde)**2)
                alpha = tf.tensordot(tf.transpose(q_l, perm=[1, 0]), y_l, axes=1)
                dt = tf.concat([dt, d_l], axis=1)
                q = tf.concat([q - tf.tensordot(b, tf.transpose(q_l, perm=[1, 0]), axes=1), tf.transpose(q_l, perm=[1, 0])], axis=0)

                # update coefficients
                x_l = tf.concat([x_l-alpha*b, alpha], axis=0)
                r = r-alpha*d_tilde
                if tf.norm(rprev) > tf.norm(r):
                    xbest = x_l
                    indices_best = tf.cast(indices, tf.int32)
                    print(indices_best)
                rprev = r

            if len(indices_best) == 0:
                indices_best = tf.cast(indices, tf.int32)
                xbest = x_l

            # fill in the coefficients
            ib_un = tf.gather(indices_best, tf.range(len(indices_best)), axis=1)
            mask_indices = [[ind, i] for ind in ib_un]
            new_coefficients_tensor = tf.SparseTensor(mask_indices, values=tf.squeeze(tf.get_static_value(xbest), axis=1), dense_shape=x.shape)
            mask = tf.sparse.add(tf.cast(tf.SparseTensor(mask_indices, values=tf.constant(-1, shape=(len(mask_indices),)), dense_shape=x.shape), tf.float32) , tf.cast(tf.ones_like(x), tf.float32))
            x = tf.sparse.add(x * mask, new_coefficients_tensor * (1 - mask))

        # x[tf.abs(x) < coeff_tolerance] = 0
        mask = tf.abs(x) >= coeff_tolerance
        mask = tf.cast(mask, tf.float32)
        new_coefficients_tensor = tf.cast(tf.SparseTensor([[0,0]], values=tf.constant(0, shape=(1,)), dense_shape=x.shape), tf.float32)
        x = tf.sparse.add(x * mask, new_coefficients_tensor * (1 - mask))

        return x

if __name__ == "__main__":
    from src.kernel_matrix_tf import KernelMatrix
    num_samples = 100
    num_classes = 10
    np.random.seed(69)
    features = np.random.random((1000, num_samples))
    np.random.seed(69)
    labels = np.random.randint(0, num_classes - 1, num_samples)
    kernel_dicts = [
        {'gaussian': {'param1': 1}},
        {'linear': {'param1': 1}},
        {'polynomial': {'param1': 1, 'param2': 2}},
    ]
    L0 = tf.convert_to_tensor(np.eye(num_classes)[labels], np.float32)
    K00 = KernelMatrix(features, features, kernel_dicts).matrix
    # Act
    W0 = RecursiveOMP().run(K00, [], L0, residual_norm=0.1)
    # Assert
