import numba as nb
import numpy as np
from .norm import _cdf, _ppf


@nb.njit(cache=True)
def _logpdf(x, s, loc, scale):
    z = (x - loc) / scale
    if z <= 0:
        return -np.inf
    c = np.sqrt(2 * np.pi)
    log_pdf = -0.5 * np.log(z) ** 2 / s ** 2 - np.log(s * z * c)
    return log_pdf - np.log(scale)


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def logpdf(x, s, loc, scale):
    """
    Return log of probability density of lognormal distribution.
    """
    return _logpdf(x, s, loc, scale)


@nb.vectorize(_signatures, cache=True)
def pdf(x, s, loc, scale):
    """
    Return probability density of lognormal distribution.
    """
    return np.exp(_logpdf(x, s, loc, scale))


@nb.vectorize(_signatures, cache=True)
def cdf(x, s, loc, scale):
    """
    Evaluate cumulative distribution function of lognormal distribution.
    """
    z = (x - loc) / scale
    if z <= 0:
        return 0.0
    return _cdf(np.log(z) / s)


@nb.vectorize(_signatures)  # cannot be cached because of call to _ppf
def ppf(p, s, loc, scale):
    """
    Return quantile of lognormal distribution for given probability.
    """
    z = np.exp(s * _ppf(p))
    return scale * z + loc
