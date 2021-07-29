import numba as nb
import numpy as np
from ._special import pdtr
from math import lgamma

_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def pdf(x, mu):
    """
    Return probability density for continuous Poisson distribution (allow non-integer k).
    """
    logp = x * np.log(mu) - lgamma(x + 1.0) - mu
    return np.exp(logp)


@nb.vectorize(_signatures)
def cdf(x, mu):
    """
    Evaluate cumulative distribution function of continuous Poisson distribution.
    """
    return pdtr(x, mu)
