"""
Normal distribution.
"""
import numpy as np
from ._special import erfinv as _erfinv
from ._util import _vectorize, _jit
from math import erf as _erf


@_jit
def _logpdf(z):
    return -0.5 * (z**2 + np.log(2 * np.pi))


@_jit
def _cdf(z):
    c = np.sqrt(0.5)
    return 0.5 * (1.0 + _erf(z * c))


@_jit(cache=False)  # cannot cache because of _erfinv
def _ppf(p):
    return np.sqrt(2) * _erfinv(2 * p - 1)


@_vectorize(3)
def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    z = (x - loc) / scale
    return _logpdf(z) - np.log(scale)


@_vectorize(3)
def pdf(x, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, loc, scale))


@_vectorize(3)
def cdf(x, loc, scale):
    """
    Return cumulative probability.
    """
    z = (x - loc) / scale
    return _cdf(z)


@_vectorize(3, cache=False)  # cannot cache this
def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    z = _ppf(p)
    return scale * z + loc
