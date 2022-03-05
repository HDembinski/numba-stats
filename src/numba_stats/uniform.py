"""
Uniform distribution.
"""
from ._util import _jit
import numpy as np


@_jit(2)
def _logpdf(x, a, w):
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        if a <= x <= a + w:
            r[i] = -np.log(w)
        else:
            r[i] = -np.inf
    return r


@_jit(2)
def _cdf(x, a, w):
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        if a <= xi:
            if xi <= a + w:
                r[i] = (xi - a) / w
            r[i] = 1
        r[i] = 0
    return r


@_jit(2)
def _ppf(p, a, w):
    return w * p + a


def logpdf(x, a, w):
    """
    Return probability density.
    """
    return _logpdf(x, a, w)


def pdf(x, a, w):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, a, w))


def cdf(x, a, w):
    """
    Return cumulative probability.
    """
    return _cdf(x, a, w)


def ppf(p, a, w):
    """
    Return quantile for given probability.
    """
    return _ppf(p, a, w)
