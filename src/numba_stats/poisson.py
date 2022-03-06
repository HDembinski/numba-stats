"""
Poisson distribution.
"""

import numpy as np
from ._special import gammaincc as _gammaincc
from math import lgamma as _lgamma
from ._util import _jit


@_jit(1)
def logpmf(k, mu):
    """
    Return log of probability mass.
    """
    T = type(mu)
    r = np.empty(len(k), T)
    for i, ki in enumerate(k):
        if mu == 0:
            r[i] = 0.0 if ki == 0 else -np.inf
        else:
            r[i] = ki * np.log(mu) - _lgamma(ki + T(1)) - mu
    return r


@_jit(1)
def pmf(k, mu):
    """
    Return probability mass.
    """
    return np.exp(logpmf(k, mu))


@_jit(1, cache=False)
def cdf(k, mu):
    """
    Evaluate cumulative distribution function.
    """
    T = type(mu)
    r = np.empty(len(k), T)
    for i, ki in enumerate(k):
        r[i] = _gammaincc(ki + T(1), mu)
    return r
