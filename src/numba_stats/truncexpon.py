import numba as nb
import numpy as np
from math import expm1 as _expm1, log1p as _log1p

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.njit
def _cdf(z):
    return -_expm1(-z)


@nb.vectorize(_signatures, cache=True)
def pdf(x, xmin, xmax, mu, sigma):
    """
    Return probability density of exponential distribution.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 0.0
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    return np.exp(-z) * sigma_inv / (_cdf(zmax) - _cdf(zmin))


@nb.vectorize(_signatures, cache=True)
def cdf(x, xmin, xmax, mu, sigma):
    """
    Evaluate cumulative distribution function of exponential distribution.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 1.0
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    p = _cdf(z)
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    return (p - pmin) / (pmax - pmin)


@nb.vectorize(_signatures, cache=True)
def ppf(p, xmin, xmax, mu, sigma):
    """
    Return quantile of exponential distribution for given probability.
    """
    sigma_inv = 1 / sigma
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    pstar = p * (pmax - pmin) + pmin
    z = -_log1p(-pstar)
    x = z * sigma + mu
    return x
