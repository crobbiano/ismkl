import time
import numpy as np
from copy import deepcopy
from numpy import ndarray
import multiprocessing
from recursive_omp import RecursiveOMP

def omp_parallel(data: ndarray, dictionary: ndarray, tau: int, tolerance: float, coeff_tolerance=0.1):
    # Solving the sparse codes for each data element (this can be parallelized since dict doesn't change)
    pool_obj = multiprocessing.Pool()
    args = [(data[:, nn], dictionary, tau, tolerance) for nn in range(data.shape[1])]
    codes_list = pool_obj.starmap(get_single_sample_codes_omp, args)
    X_sparse = np.concatenate(codes_list, axis=1)
    pool_obj.close()
    X_sparse[np.abs(X_sparse) < coeff_tolerance] = 0
    return X_sparse


def omp_parallel_chris(data: ndarray, dictionary: ndarray, tau: int, tolerance: float, coeff_tolerance=0.1):
    # Solving the sparse codes for each data element (this can be parallelized since dict doesn't change)
    pool_obj = multiprocessing.Pool()
    args = [(data[:, nn].reshape(data.shape[0], 1), dictionary, tau, tolerance) for nn in range(data.shape[1])]
    codes_list = pool_obj.starmap(get_single_sample_codes_omp_chris, args)
    X_sparse = np.concatenate(codes_list, axis=1)
    pool_obj.close()
    X_sparse[np.abs(X_sparse) < coeff_tolerance] = 0
    return X_sparse


def omp(data: ndarray, dictionary: ndarray, tau: int, tolerance: float, coeff_tolerance=0.1):
    K = dictionary.shape[1]
    X_sparse = np.zeros((K, data.shape[1]))
    # Solving the sparse codes for each data element (this can be parallelized since dict doesn't change)
    for nn in range(data.shape[1]):
        x = deepcopy(data[:, nn])  # xid[:,n] # # a single data sample to be represented in terms of atoms in D
        X_sparse[:, nn] = get_single_sample_codes_omp(x, dictionary, tau, tolerance).reshape(K,)
    X_sparse[np.abs(X_sparse) < coeff_tolerance] = 0
    return X_sparse


def get_single_sample_codes_omp(sample: ndarray, dictionary: ndarray, tau: int, tolerance: float):
    x = sample                  # single sample that we would like to find codes for in terms of dictionary
    r = deepcopy(sample)        # residual vector
    D = dictionary              # Dictionary
    K = dictionary.shape[1]     # Number of atoms in dictionary

    # Note that the o (optimal) variables are used to avoid an uncommon
    # scenario (that does occur) where a lower sparsity solution may have
    # had lower error than the final solution (with tau non zeros) but
    # wasn't low enough to break out of the coefficient solver via the error
    # tolerance. A little more memory for significantly better solutions,
    # thanks to CR for the tip (JJH)

    γₒ = 0          # this will store whatever the minimum error solution was during computation of the coefficients
    av_err = np.inf    # norm of the error vector.
    best_err = np.inf  # will store lowest error vector norm
    ii = 1      # while loop index
    DI = []     # This holds the atoms selected via OMP as its columns (it grows along 2nd dimension)
    DIGI = []   # Inverse of DI's gram matrix
    I = []      # set of indices corresponding to atoms selected in reconstruction
    Iₒ = []     # I think you get the deal with these guys now (best set of indices lul)
    X_sparse = np.zeros((K, 1))

    while (len(I) < tau) and (av_err > tolerance):
        inner_products = np.abs(np.matmul(D.transpose(), r))
        k = np.argmax(inner_products)
        dk = D[:, k].reshape(D.shape[0], 1)
        if ii == 1:
            I.append(k)
            DI = dk
            DIGI = np.array(1.0/np.matmul(DI.transpose(), DI))
        else:
            I.append(k)
            rho = np.matmul(DI.transpose(), dk)
            DI = np.concatenate((DI, dk), axis=1)
            ipk = np.linalg.norm(dk)**2
            DIGI = blockMatrixInv(DIGI, rho, rho, ipk)
        DIdag = np.matmul(DIGI, DI.transpose())
        γ = np.matmul(DIdag, x)
        r = x - np.matmul(DI, γ)
        av_err = np.linalg.norm(r)
        if av_err <= best_err:
            best_err = av_err
            γₒ = γ
            Iₒ = I
        X_sparse[I, 0] = γ
        ii += 1
    if av_err > best_err:
        X_sparse = 0 * X_sparse
        X_sparse[Iₒ, 0] = γₒ
    return X_sparse


# def blockMatrixInv(Ai: ndarray, B: ndarray, C: ndarray, D: float):
#     C = C.transpose()
#     AiB = np.matmul(Ai, B)
#     CAi = np.matmul(C, Ai)
#     DCABi = 1.0/(D - np.matmul(C, AiB))
#     AiBDCABi = np.matmul(AiB, DCABi)
#     return np.block([[Ai + np.matmul(AiBDCABi, CAi), -AiBDCABi], [np.matmul(-DCABi, CAi), DCABi]])


def get_single_sample_codes_omp_chris(y_l, d, max_iterations,  residual_norm):
    residual_norm_l = residual_norm * np.linalg.norm(y_l)
    indices = []
    K = d.shape[1]
    x_sparse = np.zeros((K, 1))
    if len(indices) == 0:
        # init vars
        r = y_l
        dt = np.expand_dims(np.asarray([]), axis=-1)
        # find best match
        idx = np.expand_dims(np.argmax(np.abs(np.dot(r.T, d))), axis=-1).astype(int)
        dt = np.concatenate([dt, d[:, idx]], axis=0)
        indices = np.concatenate([indices, idx], axis=0).astype(int)

        # Update the filter
        q = np.divide(dt.T, np.dot(dt.T, dt))
        alpha = np.dot(q, y_l)

        # Update coefficients
        x_l = alpha
        r = r - alpha * dt
    else:
        # build dictionary and filter
        dt = d[:, indices]
        # dt = self.d[ indices, :]
        q = np.dot(np.linalg.inv(np.dot(dt.T, dt)), dt.T)

        # compute coefficients and residual
        x_l = np.dot(q, y_l)
        r = y_l - np.dot(dt, x_l)

    # add atoms to d until norm of residual falls below thresholds
    xbest = x_l
    indices_best = []
    num_iteration = 1  # you already have the initial component from lines 32-57
    rprev = r
    while (np.linalg.norm(r) > residual_norm_l) and (num_iteration < max_iterations):
        num_iteration += 1
        idx = np.expand_dims(np.argmax(np.abs(np.dot(r.T, d))), axis=-1)
        d_l = d[:, idx]
        indices = np.concatenate([indices, idx], axis=0)

        # update filters
        b = np.dot(q, d_l)
        d_tilde = d_l - np.dot(dt, b)
        q_l = d_tilde / np.linalg.norm(d_tilde) ** 2
        alpha = np.dot(q_l.T, y_l)
        dt = np.concatenate([dt, d_l], axis=1)
        q = np.concatenate([q - np.dot(b, q_l.T), q_l.T], axis=0)

        # update coefficients
        x_l = np.concatenate([x_l - alpha * b, alpha], axis=0)
        r = r - alpha * d_tilde
        if np.linalg.norm(rprev) > np.linalg.norm(r):
            xbest = x_l
            indices_best = indices
        rprev = r

    if len(indices_best) == 0:
        indices_best = indices
        xbest = x_l

    # fill in the coefficients
    x_sparse[indices_best] = xbest
    return x_sparse


def blockMatrixInv(Ai: ndarray, B: ndarray, C: ndarray, D: float):
    C = C.transpose()
    AiB = np.matmul(Ai, B)
    CAi = np.matmul(C, Ai)
    DCABi = 1.0/(D - np.matmul(C, AiB))
    return np.block([[Ai + np.matmul(np.matmul(AiB, DCABi), CAi), np.matmul(-AiB, DCABi)],
                     [np.matmul(-DCABi, CAi), DCABi]])

def normalize_columns(in_matrix: ndarray):
    my_matrix = deepcopy(in_matrix)
    for col in range(my_matrix.shape[1]):
        my_matrix[:, col] = my_matrix[:, col]/np.linalg.norm(my_matrix[:, col])
    return my_matrix


if __name__ == "__main__":
    feature_dim = 100  # 20
    Dict = np.random.randn(feature_dim, 3*feature_dim)
    Dict = normalize_columns(Dict)

    # # Make our samples just be the first 4 samples then add some of the fifth sample to our 4th
    # samps = deepcopy(Dict[:, :4])
    # samps[:, 3] += Dict[:, 5]

    # Try with 40,000 samples below to see benefit of parallel approach (parallel is only faster after number of
    # samples gets very large due to overhead of setting up pool
    samps = np.random.randn(feature_dim, 10_000)

    # Set the maximum number of non-zeros allowed (sparsity factor)
    tau = 20

    # Time to compute sparse codes:
    t_0 = time.time()
    coeffs = omp_parallel(samps, Dict, tau, 1e-12, coeff_tolerance=1e-4)
    #coeffs = omp_parallel_chris(samps, Dict, tau, 1e-12, coeff_tolerance=1e-4)
    #coeffs = omp(samps, Dict, tau, 1e-12, coeff_tolerance=1e-4)
    #coeffs = RecursiveOMP(Dict, [], samps, 1e-12, coeff_tolerance=1e-4, max_iterations=tau).run_parallel()
    t_1 = time.time()

    # Verify that the reconstruction codes correspond to the columns we know we sampled
    print(coeffs)
    print(coeffs.shape)
    # Print frobenius norm of error matrix
    print(np.linalg.norm(samps - np.matmul(Dict, coeffs)))
    print(f'Method took {t_1-t_0} s')
