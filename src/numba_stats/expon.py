"""
Exponential distribution.
"""
import numpy as np
from math import expm1 as _expm1, log1p as _log1p
from ._util import _jit, _trans, _wrap


@_jit(-1)
def _cdf1(z):
    return -_expm1(-z)


@_jit(2)
def _cdf(x, loc, scale):
    z = _trans(x, loc, scale)
    for i, zi in enumerate(z):
        z[i] = _cdf1(zi)
    return z


@_jit(2)
def _ppf(p, loc, scale):
    z = np.empty_like(p)
    for i in range(len(p)):
        z[i] = -_log1p(-p[i])
    return scale * z + loc


@_jit(2)
def _logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -z - np.log(scale)


def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    return _wrap(_logpdf)(x, loc, scale)


def pdf(x, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, loc, scale))


def cdf(x, loc, scale):
    """
    Return cumulative probability.
    """
    return _wrap(_cdf)(x, loc, scale)


def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    return _wrap(_ppf)(p, loc, scale)
