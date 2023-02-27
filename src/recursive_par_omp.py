import numpy as np
from copy import deepcopy
from numpy import ndarray


def omp(data: ndarray, dictionary: ndarray, tau: int, tolerance: float):
    K = dictionary.shape[1]
    X_sparse = np.zeros((K, data.shape[1]))
    # Solving the sparse codes for each data element (this can be parallelized since dict doesn't change)
    for nn in range(data.shape[1]):
        x = deepcopy(data[:, nn])  # xid[:,n] # # a single data sample to be represented in terms of atoms in D
        r = deepcopy(x)  # residual vector
        D = dictionary  # Dictionary
        # Note that the o (optimal) variables are used to avoid an uncommon
        # scenario (that does occur) where a lower sparsity solution may have
        # had lower error than the final solution (with tau non zeros) but
        # wasn't low enough to break out of the coefficient solver via the error
        # tolerance. A litte more memory for significantly better solutions,
        # thanks to CR for the tip (JJH)
        γ = 0  # this will be the growing coefficient vector
        γₒ = 0  # this will store whatever the minimum error solution was during computation of the coefficients
        av_err = 100  # norm of the error vector.
        best_err = 100  # will store lowest error vector norm
        ii = 1  # while loop index
        DI = []  # This holds the atoms selected via OMP as its columns (it grows along 2nd dimension)
        DIGI = []  # Inverse of DI's gram matrix
        DIdag = []  # PseudoInverse of DI
        I = []  # set of indices corresponding to atoms selected in reconstruction
        Iₒ = []  # I think you get the deal with these guys now (best set of indices lul)
        while (len(I) < tau) and (av_err > tolerance):
            k = np.argmax(np.abs(np.matmul(D.transpose(), r)))
            dk = D[:, k].reshape(D.shape[0], 1)
            if ii == 1:
                I.append(k)
                DI = dk
                DIGI = np.array(np.matmul(DI.transpose(), DI)**(-1))
            else:
                I.append(k)
                rho = np.matmul(DI.transpose(), dk)
                DI = np.concatenate((DI, dk), axis=1)
                ipk = np.matmul(dk.transpose(), dk)
                DIGI = blockMatrixInv(DIGI, rho, rho, ipk)
            DIdag = np.matmul(DIGI, DI.transpose())
            γ = np.matmul(DIdag, x)
            r = x - np.matmul(DI, γ)
            av_err = np.linalg.norm(r)
            if av_err <= best_err:
                best_err = av_err
                γₒ = γ
                Iₒ = I
            X_sparse[I, nn] = γ
            ii += 1

        if av_err > best_err:
            X_sparse[I, nn] = 0 * X_sparse[I, nn]
            X_sparse[Iₒ, nn] = γₒ

    return X_sparse


def blockMatrixInv(Ai: ndarray, B: ndarray, C: ndarray, D: float):
    C = C.transpose()
    DCABi = np.linalg.inv(D - np.matmul(np.matmul(C, Ai), B))
    return np.block([[Ai + np.matmul(np.matmul(np.matmul(np.matmul(Ai, B), DCABi), C), Ai),
                        np.matmul(np.matmul(-Ai, B), DCABi)], [np.matmul(np.matmul(-DCABi, C), Ai), DCABi]])



if __name__ == "__main__":
    Dict = np.random.randn(20, 100)
    # Make our samples just be the first 4 samples then add some of the fifth sample to our 4th
    samps = deepcopy(Dict[:, :4])
    samps[:, 3] += Dict[:, 5]
    tau = 5
    coeffs = omp(samps, Dict, tau, 1e-14)
    # Verify that the reconstruction codes correspond to the columns we know we sampled
    print(coeffs)
    print(coeffs.shape)
    print(np.linalg.norm(samps - np.matmul(Dict, coeffs)))
