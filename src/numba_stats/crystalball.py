"""
Crystal Ball distribution.

The Crystal Ball distribution replaces the lower tail of a normal distribution with
a power-law tail.

https://en.wikipedia.org/wiki/Crystal_Ball_function
"""

from ._util import _jit, _vectorize
import numpy as np
from math import erf as _erf


@_jit
def _powerlaw(z, beta, m):
    assert beta > 0
    assert m > 0
    exp_beta = np.exp(-0.5 * beta**2)
    a = (m / beta) ** m * exp_beta
    b = m / beta - beta
    return a * (b - z) ** -m


@_jit
def _powerlaw_integral(z, beta, m):
    assert beta > 0
    assert m > 1
    exp_beta = np.exp(-0.5 * beta**2)
    a = (m / beta) ** m * exp_beta
    b = m / beta - beta
    m1 = m - 1
    return a * (b - z) ** -m1 / m1


@_jit
def _normal_integral(a, b):
    sqrt_half = np.sqrt(0.5)
    return sqrt_half * np.sqrt(np.pi) * (_erf(b * sqrt_half) - _erf(a * sqrt_half))


@_jit
def _density(z, beta, m):
    if z <= -beta:
        return _powerlaw(z, beta, m)
    return np.exp(-0.5 * z**2)


@_vectorize(5)
def pdf(x, beta, m, loc, scale):
    """
    Return probability density.
    """
    z = (x - loc) / scale
    dens = _density(z, beta, m)
    norm = scale * (
        _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, np.inf)
    )
    return dens / norm


@_vectorize(5)
def cdf(x, beta, m, loc, scale):
    """
    Return cumulative probability.
    """
    z = (x - loc) / scale
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, np.inf)
    if z <= -beta:
        return _powerlaw_integral(z, beta, m) / norm
    return (_powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, z)) / norm
