"""
Crystal Ball distribution.

The Crystal Ball distribution replaces the lower tail of a normal distribution with
a power-law tail.

https://en.wikipedia.org/wiki/Crystal_Ball_function
"""

from ._util import _jit, _vectorize
import numpy as np
from math import erf as _erf


@_jit(2)
def _log_powerlaw(z, beta, m):
    c = -0.5 * beta * beta
    log_a = m * np.log(m / beta) + c
    b = m / beta - beta
    return log_a - m * np.log(b - z)


@_jit(2)
def _powerlaw_integral(z, beta, m):
    exp_beta = np.exp(-0.5 * beta**2)
    a = (m / beta) ** m * exp_beta
    b = m / beta - beta
    m1 = m - 1
    return a * (b - z) ** -m1 / m1


@_jit(-2)
def _normal_integral(a, b):
    sqrt_half = np.sqrt(0.5)
    return sqrt_half * np.sqrt(np.pi) * (_erf(b * sqrt_half) - _erf(a * sqrt_half))


@_jit(2)
def _log_density(z, beta, m):
    if z < -beta:
        return _log_powerlaw(z, beta, m)
    return -0.5 * z**2


@_jit(4)
def _logpdf(x, beta, m, loc, scale):
    z = (x - loc) / scale
    log_dens = _log_density(z, beta, m)
    norm = scale * (
        _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, np.inf)
    )
    return log_dens - np.log(norm)


@_vectorize(5)
def logpdf(x, beta, m, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(x, beta, m, loc, scale)


@_vectorize(5)
def pdf(x, beta, m, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, beta, m, loc, scale))


@_vectorize(5)
def cdf(x, beta, m, loc, scale):
    """
    Return cumulative probability.
    """
    z = (x - loc) / scale
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, np.inf)
    if z < -beta:
        return _powerlaw_integral(z, beta, m) / norm
    return (_powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, z)) / norm
