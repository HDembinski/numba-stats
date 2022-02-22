"""
Exponential distribution.
"""
import numpy as np
from math import expm1 as _expm1, log1p as _log1p
from ._util import _jit, _vectorize


@_jit
def _cdf(z):
    return -_expm1(-z)


@_jit
def _ppf(p):
    return -_log1p(-p)


@_jit
def _logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -z - np.log(scale)


@_vectorize(3)
def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(x, loc, scale)


@_vectorize(3)
def pdf(x, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, loc, scale))


@_vectorize(3)
def cdf(x, loc, scale):
    """
    Return cumulative probability.
    """
    z = (x - loc) / scale
    return _cdf(z)


@_vectorize(3)
def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    z = _ppf(p)
    x = z * scale + loc
    return x
