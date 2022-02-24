"""
Truncated exponential distribution.
"""
import numpy as np
from ._util import _vectorize
from .expon import _cdf, _ppf


@_vectorize(5)
def logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
    if x < xmin:
        return -np.inf
    elif x > xmax:
        return -np.inf
    scale_inv = 1 / scale
    z = (x - loc) * scale_inv
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    return -z + np.log(scale_inv / (_cdf(zmax) - _cdf(zmin)))


@_vectorize(5)
def pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, xmin, xmax, loc, scale))


@_vectorize(5)
def cdf(x, xmin, xmax, loc, scale):
    """
    Return cumulative probability.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 1.0
    scale_inv = 1 / scale
    z = (x - loc) * scale_inv
    p = _cdf(z)
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    return (p - pmin) / (pmax - pmin)


@_vectorize(5)
def ppf(p, xmin, xmax, loc, scale):
    """
    Return quantile for given probability.
    """
    scale_inv = 1 / scale
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    pstar = p * (pmax - pmin) + pmin
    z = _ppf(pstar)
    x = z * scale + loc
    return x
