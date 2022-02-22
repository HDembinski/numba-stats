"""
Uniform distribution.
"""
from ._util import _vectorize
import numpy as np


@_vectorize(3)
def logpdf(x, a, w):
    """
    Return probability density.
    """
    if a <= x <= a + w:
        return -np.log(w)
    return -np.inf


@_vectorize(3)
def pdf(x, a, w):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, a, w))


@_vectorize(3)
def cdf(x, a, w):
    """
    Return cumulative probability.
    """
    if a <= x:
        if x <= a + w:
            return (x - a) / w
        return 1
    return 0


@_vectorize(3)
def ppf(p, a, w):
    """
    Return quantile for given probability.
    """
    return w * p + a
