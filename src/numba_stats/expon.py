import numba as nb
import numpy as np
from math import expm1, log1p

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, mu, sigma):
    """
    Return probability density of exponential distribution.
    """
    z = (x - mu) / sigma
    return np.exp(-z) / sigma


@nb.vectorize(_signatures, cache=True)
def cdf(x, mu, sigma):
    """
    Evaluate cumulative distribution function of exponential distribution.
    """
    z = (x - mu) / sigma
    return -expm1(-z)


@nb.vectorize(_signatures, cache=True)
def ppf(p, mu, sigma):
    """
    Return quantile of exponential distribution for given probability.
    """
    z = -log1p(-p)
    x = z * sigma + mu
    return x
