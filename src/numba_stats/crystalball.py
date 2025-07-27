"""
Crystal Ball distribution.

The Crystal Ball distribution replaces the lower tail of a normal distribution with
a power-law tail.

https://en.wikipedia.org/wiki/Crystal_Ball_function

See Also
--------
scipy.stats.crystalball: Scipy equivalent.
"""

from . import norm as _norm
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


@_jit(3, narg=0)
def _log_powerlaw(z, beta, m):
    T = type(beta)
    c = -T(0.5) * beta * beta
    log_a = m * np.log(m / beta) + c
    b = m / beta - beta
    return log_a - m * np.log(b - z)


@_jit(3, narg=0)
def _powerlaw_integral(z, beta, m):
    T = type(beta)
    log_a = m * np.log(m / beta) - T(0.5) * beta * beta
    b = m / beta - beta
    m1 = m - T(1)
    return np.exp(log_a - m1 * np.log(b - z) - np.log(m1))


@_jit(2, narg=0)
def _normal_integral(a, b):
    T = type(a)
    sqrt_half = np.sqrt(T(0.5))
    return sqrt_half * np.sqrt(T(np.pi)) * (_erf(b * sqrt_half) - _erf(a * sqrt_half))


@_jit(3, narg=0)
def _powerlaw_ppf(p, beta, m):
    T = type(beta)
    log_a = m * np.log(m / beta) - T(0.5) * beta * beta
    b = m / beta - beta
    m1 = m - T(1)
    log_term = (-T(1) / m1) * (np.log(p) + np.log(m1) - log_a)
    term = np.exp(log_term)
    return b - term


@_jit(2, narg=0, cache=False)
def _normal_ppf(p, a):
    T = type(a)
    sqrt_2pi = np.sqrt(T(2) * np.pi)
    cdf_a = _norm._cdf1(a)
    cdf_target = cdf_a + p / sqrt_2pi
    return _norm._ppf1(cdf_target)


@_jit(3, narg=0)
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


@_jit(4, cache=False)
def _ppf(p, beta, m, loc, scale):
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(
        -beta, type(beta)(np.inf)
    )
    pbeta = _powerlaw_integral(-beta, beta, m) / norm
    r = np.empty_like(p)
    for i in _prange(len(r)):
        if p[i] < pbeta:
            r[i] = _powerlaw_ppf(p[i] * norm, beta, m)
        else:
            r[i] = _normal_ppf((p[i] - pbeta) * norm, -beta)
    return scale * r + loc


_generate_wrappers(globals())
