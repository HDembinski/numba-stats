import numba as nb
import numpy as np
from math import erf as _erf


@nb.njit(cache=True)
def _constants(beta, m):
    assert beta > 0
    assert m > 1
    exp_beta = np.exp(-0.5 * beta**2)
    c = m / (beta * (m - 1.0)) * exp_beta
    d = np.sqrt(0.5 * np.pi) * (1.0 + _erf(beta * np.sqrt(0.5)))
    return exp_beta, 1.0 / (c + d)


@nb.njit(cache=True)
def _pdf(z, beta, m):
    exp_beta, nf = _constants(beta, m)

    if z <= -beta:
        a = (m / beta) ** m * exp_beta
        b = m / beta - beta
        return nf * a * (b - z) ** -m
    return nf * np.exp(-0.5 * z**2)


@nb.njit(cache=True)
def _cdf(z, beta, m):
    exp_beta, nf = _constants(beta, m)

    if z <= -beta:
        return nf * (
            (m / beta) ** m * exp_beta * (m / beta - beta - z) ** (1.0 - m) / (m - 1.0)
        )
    return nf * (
        (m / beta) * exp_beta / (m - 1.0)
        + np.sqrt(0.5 * np.pi) * (_erf(z * np.sqrt(0.5)) - _erf(-beta * np.sqrt(0.5)))
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
