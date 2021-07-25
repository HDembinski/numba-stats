import numpy as np
import numba as nb

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def bernstein_density(x, beta, xmin, xmax):
    scale = 1.0 / (xmax - xmin)
    z = (x - xmin) * scale
    if len(beta) == 0:
        return scale

    # De Casteljau algorithm, numerically stable
    n = len(beta)
    az = 1.0 - z
    beta = beta.copy()
    for j in range(1, n):
        for i in range(n - j):
            beta[i] = beta[i] * az + beta[i + 1] * z
    return beta[0] * scale * (n + 1)


@nb.vectorize(_signatures)
def bernstein_scaled_cdf(x, beta, xmin, xmax):
    beta1 = np.empty(len(beta) + 1, dtype=beta.dtype)
    beta1[1:] = beta
    beta1[0] = 0
    return bernstein_density(x, beta1, xmin, xmax)
