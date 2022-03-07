"""
Uniform distribution.
"""
from ._util import _jit, _generate_wrappers
import numpy as np


@_jit(2)
def _logpdf(x, a, w):
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
def _pdf(x, a, w):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, a, w))


@_jit(2)
def _cdf(x, a, w):
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
def _ppf(p, a, w):
    """
    Return quantile for given probability.
    """
    return w * p + a


_generate_wrappers(globals())
