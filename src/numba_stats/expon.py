"""
Exponential distribution.
"""
import numpy as np
from math import expm1 as _expm1, log1p as _log1p
from ._util import _jit, _trans


@_jit(-1)
def _cdf(z):
    return -_expm1(-z)


@_jit(-1)
def _ppf(p):
    return -_log1p(-p)


@_jit(2)
def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    z = (x - loc) / scale
    return -z - np.log(scale)


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
    z = _trans(x, loc, scale)
    for i, zi in enumerate(z):
        z[i] = _cdf(zi)
    return z


@_jit(2)
def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    z = np.empty_like(p)
    for i, pi in enumerate(p):
        z[i] = _ppf(pi)
    return scale * z + loc
