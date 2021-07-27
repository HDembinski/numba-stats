import numpy as np
import numba as nb


_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.njit(_signatures)
def _de_castlejau(z, beta, res):
    # De Casteljau algorithm, numerically stable
    n = len(beta)
    if n == 0:
        res[:] = np.nan
    else:
        betai = np.empty_like(beta)
        for iz, zi in enumerate(z):
            azi = 1.0 - zi
            betai[:] = beta
            for j in range(1, n):
                for k in range(n - j):
                    betai[k] = betai[k] * azi + betai[k + 1] * zi
            res[iz] = betai[0]


_signatures = [
    nb.float32[:](nb.float32[:], nb.float32[:]),
    nb.float64[:](nb.float64[:], nb.float64[:]),
]


@nb.njit(_signatures)
def _beta_int(beta, inverse_scale):
    n = len(beta)
    r = np.zeros(n + 1, dtype=beta.dtype)
    for j in range(1, n + 1):
        for k in range(j):
            r[j] += beta[k]
    inverse_scale /= n * (n + 1)
    r *= inverse_scale
    return r


_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.guvectorize(_signatures, "(),(n),(),()->()")
def bernstein_density(x, beta, xmin, xmax, res):
    inverse_scale = np.ones_like(xmax) / (xmax - xmin)
    z = x.copy()
    z -= xmin
    z *= inverse_scale
    inverse_scale /= len(beta) + 1
    beta_scaled = beta.copy()
    beta_scaled *= inverse_scale
    _de_castlejau(z, beta_scaled, res)


@nb.guvectorize(_signatures, "(),(n),(),()->()")
def bernstein_scaled_cdf(x, beta, xmin, xmax, res):
    inverse_scale = np.ones_like(xmax) / (xmax - xmin)
    z = x.copy()
    z -= xmin
    z *= inverse_scale
    beta = _beta_int(beta, inverse_scale)
    _de_castlejau(z, beta, res)
