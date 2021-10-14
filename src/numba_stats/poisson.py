import numba as nb
import numpy as np
from ._special import gammaincc as _gammaincc
from math import lgamma as _lgamma

_signatures = [
    nb.float32(nb.int32, nb.float32),
    nb.float64(nb.intp, nb.float64),
]


@nb.vectorize(_signatures)
def pmf(k, mu):
    """
    Return probability mass for Poisson distribution.
    """
    if mu == 0:
        return 1.0 if k == 0 else 0.0
    logp = k * np.log(mu) - _lgamma(k + 1.0) - mu
    return np.exp(logp)


@nb.vectorize(_signatures)
def cdf(k, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    return _gammaincc(k + 1, mu)
