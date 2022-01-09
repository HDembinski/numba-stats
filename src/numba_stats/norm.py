import numba as nb
import numpy as np
from ._special import erfinv as _erfinv
from math import erf as _erf


@nb.njit(cache=True)
def _logpdf(z):
    return -0.5 * (z ** 2 + np.log(2 * np.pi))


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
def logpdf(x, mu, sigma):
    """
    Return log of probability density of normal distribution.
    """
    z = (x - mu) / sigma
    return _logpdf(z) - np.log(sigma)


@nb.vectorize(_signatures, cache=True)
def pdf(x, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    # cannot call logpdf directly here, because nb.vectorize does not generate
    # inlinable code
    z = (x - mu) / sigma
    return np.exp(_logpdf(z)) / sigma


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
