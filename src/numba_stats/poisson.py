"""
Poisson distribution.
"""

import numba as nb
import numpy as np
from ._special import gammaincc as _gammaincc
from math import lgamma as _lgamma
from ._util import _jit, _wrap

_signatures = [
    nb.float32[:](nb.int32[:], nb.float32),
    nb.float64[:](nb.intp[:], nb.float64),
]


@_jit(_signatures)
def _logpmf(k, mu):
    T = type(mu)
    r = np.empty(len(k), T)
    for i, ki in enumerate(k):
        if mu == 0:
            r[i] = 0.0 if ki == 0 else -np.inf
        else:
            r[i] = ki * np.log(mu) - _lgamma(ki + T(1)) - mu
    return r


@_jit(_signatures)
def _cdf(k, mu):
    T = type(mu)
    r = np.empty(len(k), T)
    for i, ki in enumerate(k):
        r[i] = _gammaincc(ki + T(1), mu)
    return r


def logpmf(k, mu):
    """
    Return log of probability mass.
    """
    return _wrap(_logpmf)(k, mu)


def pmf(k, mu):
    """
    Return probability mass.
    """
    return np.exp(logpmf(k, mu))


def cdf(k, mu):
    """
    Evaluate cumulative distribution function.
    """
    return _wrap(_cdf)(k, mu)
