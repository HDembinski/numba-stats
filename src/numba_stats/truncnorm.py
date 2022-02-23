"""
Truncated normal distribution.
"""

import numpy as np
from .norm import _logpdf as _norm_logpdf, _cdf, _ppf
from ._util import _jit, _vectorize


@_jit
def _logpdf(z, zmin, zmax):
    if z < zmin or z > zmax:
        return -np.inf
    return _norm_logpdf(z) - np.log(_cdf(zmax) - _cdf(zmin))


@_vectorize(5)
def logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
    scale_inv = 1 / scale
    z = (x - loc) * scale_inv
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    return _logpdf(z, zmin, zmax) + np.log(scale_inv)


@_vectorize(5)
def pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density.
    """
    scale_inv = 1 / scale
    z = (x - loc) * scale_inv
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    return np.exp(_logpdf(z, zmin, zmax)) * scale_inv


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
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    return (_cdf(z) - pmin) / (pmax - pmin)


@_vectorize(5, cache=False)
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
    return scale * z + loc
