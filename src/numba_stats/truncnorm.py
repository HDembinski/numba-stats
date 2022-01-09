import numba as nb
import numpy as np
from .norm import _logpdf as _norm_logpdf, _cdf, _ppf

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.njit(cache=True)
def _logpdf(z, zmin, zmax):
    if z < zmin or z > zmax:
        return -np.inf
    return _norm_logpdf(z) - np.log(_cdf(zmax) - _cdf(zmin))


@nb.vectorize(_signatures, cache=True)
def logpdf(x, xmin, xmax, mu, sigma):
    """
    Return log of probability density of normal distribution.
    """
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    return _logpdf(z, zmin, zmax) + np.log(sigma_inv)


@nb.vectorize(_signatures, cache=True)
def pdf(x, xmin, xmax, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    return np.exp(_logpdf(z, zmin, zmax)) * sigma_inv


@nb.vectorize(_signatures, cache=True)
def cdf(x, xmin, xmax, mu, sigma):
    """
    Evaluate cumulative distribution function of normal distribution.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 1.0
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    return (_cdf(z) - pmin) / (pmax - pmin)


@nb.vectorize(_signatures)
def ppf(p, xmin, xmax, mu, sigma):
    """
    Return quantile of normal distribution for given probability.
    """
    sigma_inv = 1 / sigma
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    pstar = p * (pmax - pmin) + pmin
    z = _ppf(pstar)
    return sigma * z + mu
