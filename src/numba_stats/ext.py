import numpy as np
import numba as nb

_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.guvectorize(_signatures, "(n),(m)->(n)")
def _de_castlejau(z, beta, res):
    # De Casteljau algorithm, numerically stable
    n = len(beta)

    if n == 0:
        res[:] = 1.0
    else:
        betai = np.empty_like(beta)
        for iz, zi in enumerate(z):
            azi = 1.0 - zi
            betai[:] = beta
            for j in range(1, n):
                for k in range(n - j):
                    betai[k] = betai[k] * azi + betai[k + 1] * zi
            res[iz] = betai[0]


def bernstein_density(x, beta, xmin, xmax):
    scale = 1.0 / (xmax - xmin)
    x = np.atleast_1d(x)
    z = (x - xmin) * scale
    res = np.empty_like(z)
    beta = np.atleast_1d(beta)
    _de_castlejau(z, beta, res)
    return res * scale / (len(beta) + 1)


def bernstein_scaled_cdf(x, beta, xmin, xmax):
    beta = np.atleast_1d(beta)
    beta1 = np.empty(len(beta) + 1, dtype=beta.dtype)
    beta1[1:] = beta
    beta1[0] = 0
    scale = 1.0 / (xmax - xmin)
    x = np.atleast_1d(x)
    z = (x - xmin) * scale
    res = np.empty_like(z)
    _de_castlejau(z, beta1, res)
    return res * scale / len(beta1)
