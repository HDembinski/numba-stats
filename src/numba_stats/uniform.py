"""
Uniform distribution.
"""
from ._util import _jit
import numpy as np


@_jit(2)
def logpdf(x, a, w):
    """
    Return log of probability density.
    """
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        if a <= xi <= a + w:
            r[i] = -np.log(w)
        else:
            r[i] = -np.inf
    return r


@_jit(2)
def pdf(x, a, w):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, a, w))


@_jit(2)
def cdf(x, a, w):
    """
    Return cumulative probability.
    """
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        if a <= xi:
            if xi <= a + w:
                r[i] = (xi - a) / w
            else:
                r[i] = 1
        else:
            r[i] = 0
    return r


@_jit(2)
def ppf(p, a, w):
    """
    Return quantile for given probability.
    """
    return w * p + a
