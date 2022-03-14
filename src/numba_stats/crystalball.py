"""
Crystal Ball distribution.

The Crystal Ball distribution replaces the lower tail of a normal distribution with
a power-law tail.

https://en.wikipedia.org/wiki/Crystal_Ball_function

See Also
--------
scipy.stats.crystalball: Scipy equivalent.
"""
from ._util import _jit, _trans, _generate_wrappers, _prange
import numpy as np
from math import erf as _erf

_doc_par = """
x : Array-like
    Random variate.
beta : float
    Distance from the mode in units of standard deviations where the Crystal
    Ball turns from a gaussian into a power law.
m : float
    Absolute value of the slope of the powerlaw tail. Must be large than 1.
"""


@_jit(-3)
def _log_powerlaw(z, beta, m):
    T = type(beta)
    c = -T(0.5) * beta * beta
    log_a = m * np.log(m / beta) + c
    b = m / beta - beta
    return log_a - m * np.log(b - z)


@_jit(-3)
def _powerlaw_integral(z, beta, m):
    T = type(beta)
    exp_beta = np.exp(-T(0.5) * beta * beta)
    a = (m / beta) ** m * exp_beta
    b = m / beta - beta
    m1 = m - type(m)(1)
    return a * (b - z) ** -m1 / m1


@_jit(-2)
def _normal_integral(a, b):
    T = type(a)
    sqrt_half = np.sqrt(T(0.5))
    return sqrt_half * np.sqrt(T(np.pi)) * (_erf(b * sqrt_half) - _erf(a * sqrt_half))


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
    for i in _prange(len(z)):
        z[i] = _log_density(z[i], beta, m) - c
    return z


@_jit(4)
def _pdf(x, beta, m, loc, scale):
    return np.exp(_logpdf(x, beta, m, loc, scale))


@_jit(4)
def _cdf(x, beta, m, loc, scale):
    z = _trans(x, loc, scale)
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(
        -beta, type(beta)(np.inf)
    )
    for i in _prange(len(z)):
        if z[i] < -beta:
            z[i] = _powerlaw_integral(z[i], beta, m) / norm
        else:
            z[i] = (
                _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, z[i])
            ) / norm
    return z


_generate_wrappers(globals())
