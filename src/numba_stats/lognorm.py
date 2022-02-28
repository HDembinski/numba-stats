"""
Lognormal distribution.
"""
import numpy as np
from .norm import _cdf, _ppf
from ._util import _jit, _vectorize


# has to be separate to avoid a warning
@_jit
def _logpdf(x, s, loc, scale):
    z = (x - loc) / scale
    if z <= 0:
        return -np.inf
    c = np.sqrt(2 * np.pi)
    log_pdf = -0.5 * np.log(z) ** 2 / s**2 - np.log(s * z * c)
    return log_pdf - np.log(scale)


@_vectorize(4)
def logpdf(x, s, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(x, s, loc, scale)


@_vectorize(4)
def pdf(x, s, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, s, loc, scale))


@_vectorize(4)
def cdf(x, s, loc, scale):
    """
    Return cumulative probability.
    """
    z = (x - loc) / scale
    if z <= 0:
        return 0.0
    return _cdf(np.log(z) / s)


@_vectorize(4, cache=False)  # no cache because of _ppf
def ppf(p, s, loc, scale):
    """
    Return quantile for given probability.
    """
    z = np.exp(s * _ppf(p))
    return scale * z + loc
