"""
Student's t distribution.
"""
import numpy as np
from ._special import stdtr as _stdtr, stdtrit as _stdtrit
from ._util import _jit, _trans, _wrap
from math import lgamma as _lgamma


@_jit(3, cache=False)
def _logpdf(x, df, loc, scale):
    T = type(df)
    z = _trans(x, loc, scale)
    k = T(0.5) * (df + T(1))
    c = _lgamma(k) - _lgamma(T(0.5) * df)
    c -= T(0.5) * np.log(df * T(np.pi))
    c -= np.log(scale)
    for i, zi in enumerate(z):
        z[i] = -k * np.log(T(1) + (zi * zi) / df) + c
    return z


@_jit(3, cache=False)
def _cdf(x, df, loc, scale):
    z = _trans(x, loc, scale)
    for i, zi in enumerate(z):
        z[i] = _stdtr(df, zi)
    return z


@_jit(3, cache=False)
def _ppf(p, df, loc, scale):
    T = type(df)
    r = np.empty_like(p)
    for i, pi in enumerate(p):
        if pi == 0:
            r[i] = -T(np.inf)
        elif pi == 1:
            r[i] = T(np.inf)
        else:
            r[i] = scale * _stdtrit(df, pi) + loc
    return r


def logpdf(x, df, loc, scale):
    """
    Return probability density.
    """
    return _wrap(_logpdf)(x, df, loc, scale)


def pdf(x, df, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, df, loc, scale))


def cdf(x, df, loc, scale):
    """
    Return cumulative probability.
    """
    return _wrap(_cdf)(x, df, loc, scale)


def ppf(p, df, loc, scale):
    """
    Return quantile for given probability.
    """
    return _wrap(_ppf)(p, df, loc, scale)
