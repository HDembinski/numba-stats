"""
Crystal Ball distribution.

The Crystal Ball distribution replaces the lower tail of a normal distribution with
a power-law tail.

https://en.wikipedia.org/wiki/Crystal_Ball_function
"""

from ._util import _jit, _trans
import numpy as np
from math import erf as _erf


@_jit(-3)
def _log_powerlaw(z, beta, m):
    c = -type(beta)(0.5) * beta * beta
    log_a = m * np.log(m / beta) + c
    b = m / beta - beta
    return log_a - m * np.log(b - z)


@_jit(-3)
def _powerlaw_integral(z, beta, m):
    exp_beta = np.exp(-type(beta)(0.5) * beta * beta)
    a = (m / beta) ** m * exp_beta
    b = m / beta - beta
    m1 = m - type(m)(1)
    return a * (b - z) ** -m1 / m1


@_jit(-2)
def _normal_integral(a, b):
    sqrt_half = np.sqrt(type(a)(0.5))
    return (
        sqrt_half
        * np.sqrt(type(a)(np.pi))
        * (_erf(b * sqrt_half) - _erf(a * sqrt_half))
    )


@_jit(-3)
def _log_density(z, beta, m):
    if z < -beta:
        return _log_powerlaw(z, beta, m)
    return -0.5 * z * z


@_jit(4)
def _logpdf(x, beta, m, loc, scale):
    z = _trans(x, loc, scale)
    norm = scale * (
        _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, type(beta)(np.inf))
    )
    c = np.log(norm)
    for i, zi in enumerate(z):
        z[i] = _log_density(zi, beta, m) - c
    return z


@_jit(4)
def _cdf(x, beta, m, loc, scale):
    z = _trans(x, loc, scale)
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(
        -beta, type(beta)(np.inf)
    )
    for i, zi in enumerate(z):
        if zi < -beta:
            z[i] = _powerlaw_integral(zi, beta, m) / norm
        else:
            z[i] = (
                _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, zi)
            ) / norm
    return z


def logpdf(x, beta, m, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(x, beta, m, loc, scale)


def pdf(x, beta, m, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, beta, m, loc, scale))


def cdf(x, beta, m, loc, scale):
    """
    Return cumulative probability.
    """
    return _cdf(x, beta, m, loc, scale)
