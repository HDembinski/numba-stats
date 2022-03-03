"""
Student's t distribution.
"""
import numpy as np
from ._special import stdtr as _cdf, stdtrit as _ppf
from ._util import _vectorize
from math import lgamma as _lgamma


@_vectorize(4, cache=False)
def logpdf(x, df, loc, scale):
    """
    Return probability density.
    """
    z = (x - loc) / scale
    k = 0.5 * (df + 1)
    logp = _lgamma(k) - _lgamma(0.5 * df)
    logp -= 0.5 * np.log(df * np.pi) + k * np.log(1 + (z**2) / df) + np.log(scale)
    return logp


@_vectorize(4, cache=False)
def pdf(x, df, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, df, loc, scale))


@_vectorize(4, cache=False)
def cdf(x, df, loc, scale):
    """
    Return cumulative probability.
    """
    z = (x - loc) / scale
    return _cdf(df, z)


@_vectorize(4, cache=False)
def ppf(p, df, loc, scale):
    """
    Return quantile for given probability.
    """
    if p == 0:
        return -np.inf
    elif p == 1:
        return np.inf
    z = _ppf(df, p)
    return scale * z + loc
