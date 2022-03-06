"""
Normal distribution.
"""
import numpy as np
from ._special import erfinv as _erfinv
from ._util import _jit, _trans
from math import erf as _erf


@_jit(-1)
def _logpdf(z):
    T = type(z)
    return -T(0.5) * (z * z + T(np.log(2 * np.pi)))


@_jit(-1)
def _cdf(z):
    T = type(z)
    c = T(np.sqrt(0.5))
    return T(0.5) * (T(1.0) + _erf(z * c))


@_jit(-1, cache=False)  # cannot cache because of _erfinv
def _ppf(p):
    T = type(p)
    return T(np.sqrt(2)) * _erfinv(T(2) * p - T(1))


@_jit(2)
def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        r[i] = _logpdf(ri) - np.log(scale)
    return r


@_jit(2)
def pdf(x, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, loc, scale))


@_jit(2)
def cdf(x, loc, scale):
    """
    Return cumulative probability.
    """
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        r[i] = _cdf(ri)
    return r


@_jit(2, cache=False)
def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    r = np.empty_like(p)
    for i in range(len(p)):
        r[i] = scale * _ppf(p[i]) + loc
    return r
