"""
Poisson distribution.
"""

import numba as nb
import numpy as np
from ._special import gammaincc as _gammaincc
from math import lgamma as _lgamma

_signatures = [
    nb.float32(nb.int32, nb.float32),
    nb.float64(nb.intp, nb.float64),
]


@nb.vectorize(_signatures)
def logpmf(k, mu):
    """
    Return log of probability mass.
    """
    if mu == 0:
        return 0.0 if k == 0 else -np.inf
    return k * np.log(mu) - _lgamma(k + 1.0) - mu


@nb.vectorize(_signatures)
def pmf(k, mu):
    """
    Return probability mass.
    """
    return np.exp(logpmf(k, mu))


@nb.vectorize(_signatures)
def cdf(k, mu):
    """
    Evaluate cumulative distribution function.
    """
    return _gammaincc(k + 1, mu)
