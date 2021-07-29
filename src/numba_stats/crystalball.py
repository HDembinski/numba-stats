import numba as nb
import numpy as np
from math import erf


@nb.njit(cache=True)
def _pdf(z, beta, m):
    assert beta > 0
    assert m > 1

    exp_beta = np.exp(-0.5 * beta ** 2)

    c = m / (beta * (m - 1.0)) * exp_beta
    # d = _norm_cdf(-beta) * np.sqrt(2 * np.pi)
    d = np.sqrt(0.5 * np.pi) * (1.0 + erf(beta * np.sqrt(0.5)))
    n = 1.0 / (c + d)

    if z <= -beta:
        a = (m / beta) ** m * exp_beta
        b = m / beta - beta
        return n * a * (b - z) ** -m
    return n * np.exp(-0.5 * z ** 2)


@nb.njit(cache=True)
def _cdf(z, beta, m):
    exp_beta = np.exp(-0.5 * beta ** 2)
    c = m / (beta * (m - 1.0)) * exp_beta
    d = np.sqrt(0.5 * np.pi) * (1.0 + erf(beta * np.sqrt(0.5)))
    n = 1.0 / (c + d)

    if z <= -beta:
        return n * (
            (m / beta) ** m * exp_beta * (m / beta - beta - z) ** (1.0 - m) / (m - 1.0)
        )
    return n * (
        (m / beta) * exp_beta / (m - 1.0)
        + np.sqrt(0.5 * np.pi) * (erf(z * np.sqrt(0.5)) - erf(-beta * np.sqrt(0.5)))
    )


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, beta, m, loc, scale):
    z = (x - loc) / scale
    return _pdf(z, beta, m) / scale


@nb.vectorize(_signatures, cache=True)
def cdf(x, beta, m, loc, scale):
    z = (x - loc) / scale
    return _cdf(z, beta, m)
