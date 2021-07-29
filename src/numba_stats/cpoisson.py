import numba as nb
import numpy as np
from ._special import xlogy, gammaln, pdtr

_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def pdf(k, mu):
    """
    Return probability mass for Poisson distribution (allow non-integer k)
    """
    logp = xlogy(k, mu) - gammaln(k + 1.0) - mu
    return np.exp(logp)


_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def cdf(x, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    k = np.floor(x)  # TODO is this correct?
    return pdtr(k, mu)
