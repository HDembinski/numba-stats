import numba as nb
import numpy as np
from ._special import erfinv as _erfinv
from math import erf as _erf


@nb.njit(cache=True)
def _pdf(z):
    c = 1.0 / np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) * c

@nb.njit(cache=True)
def _logpdf(z):
    c = 1.0 / np.sqrt(2 * np.pi)
    return (-0.5 * z ** 2) + np.log(c)


@nb.njit(cache=True)
def _cdf(z):
    c = np.sqrt(0.5)
    return 0.5 * (1.0 + _erf(z * c))


@nb.njit
def _ppf(p):
    return np.sqrt(2) * _erfinv(2 * p - 1)


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    z = (x - mu) / sigma
    return _pdf(z) / sigma


@nb.vectorize(_signatures, cache=True)
def cdf(x, mu, sigma):
    """
    Evaluate cumulative distribution function of normal distribution.
    """
    z = (x - mu) / sigma
    return _cdf(z)


@nb.vectorize(_signatures)
def ppf(p, mu, sigma):
    """
    Return quantile of normal distribution for given probability.
    """
    z = _ppf(p)
    return sigma * z + mu


@nb.vectorize(_signatures, cache=True)
def logpdf(x, mu, sigma):
    """
    Return log of probability density of normal distribution.
    """
    z = (x - mu) / sigma
    return _logpdf(z) - np.log(sigma)
