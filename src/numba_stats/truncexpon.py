import numpy as np
from ._util import _vectorize
from .expon import _cdf, _ppf


@_vectorize(5)
def pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density of exponential distribution.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 0.0
    scale_inv = 1 / scale
    z = (x - loc) * scale_inv
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    return np.exp(-z) * scale_inv / (_cdf(zmax) - _cdf(zmin))


@_vectorize(5)
def cdf(x, xmin, xmax, loc, scale):
    """
    Return cumulative probability of truncated exponential distribution.
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
    Return quantile of exponential distribution for given probability.
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
