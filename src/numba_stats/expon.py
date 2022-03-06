"""
Exponential distribution.
"""
import numpy as np
from math import expm1 as _expm1, log1p as _log1p
from ._util import _jit, _trans, _Floats, _cast


@_jit(-1)
def _cdf1(z):
    return -_expm1(-z)


@_jit([(T[:],) for T in _Floats])
def _cdf_inplace(z):
    for i, zi in enumerate(z):
        z[i] = _cdf1(zi)


@_jit(0)
def _ppf(p):
    z = np.empty_like(p)
    for i in range(len(p)):
        z[i] = -_log1p(-p[i])
    return z


@_jit(2)
def _logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -z - np.log(scale)


def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(_cast(x), loc, scale)


def pdf(x, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(_cast(x), loc, scale))


def cdf(x, loc, scale):
    """
    Return cumulative probability.
    """
    z = _trans(_cast(x), loc, scale)
    _cdf_inplace(z)
    return z


def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    z = _ppf(_cast(p))
    x = z * scale + loc
    return x
