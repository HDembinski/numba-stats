import numba as nb
import numpy as np
from ._special import stdtr as _cdf, stdtrit as _ppf
from math import lgamma as _lgamma

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, df, mu, sigma):
    """
    Return probability density of student's distribution.
    """
    z = (x - mu) / sigma
    k = 0.5 * (df + 1)
    p = np.exp(_lgamma(k) - _lgamma(0.5 * df))
    p /= np.sqrt(df * np.pi) * (1 + (z ** 2) / df) ** k
    return p / sigma


@nb.vectorize(_signatures)
def cdf(x, df, mu, sigma):
    """
    Evaluate cumulative distribution function of student's distribution.
    """
    z = (x - mu) / sigma
    return _cdf(df, z)


@nb.vectorize(_signatures)
def ppf(p, df, mu, sigma):
    """
    Return quantile of student's distribution for given probability.
    """
    if p == 0:
        return -np.inf
    elif p == 1:
        return np.inf
    z = _ppf(df, p)
    return sigma * z + mu
