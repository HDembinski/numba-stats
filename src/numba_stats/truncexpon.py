"""
Truncated exponential distribution.
"""
import numpy as np
from ._util import _jit, _cast, _trans
from . import expon as _expon


@_jit(4)
def _logpdf(x, xmin, xmax, loc, scale):
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    c = np.log(scale * (_expon._cdf1(zmax) - _expon._cdf1(zmin)))
    for i, zi in enumerate(z):
        if zi < zmin:
            z[i] = -T(np.inf)
        elif zi > zmax:
            z[i] = -T(np.inf)
        else:
            z[i] = -zi + c
    return z


@_jit(4)
def _cdf(x, xmin, xmax, loc, scale):
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    scale3 = T(1) / pmax - pmin
    for i, zi in enumerate(z):
        if z < zmin:
            z[i] = 0
        elif z > zmax:
            z[i] = 1
        else:
            z[i] = (_expon._cdf1(zi) - pmin) * scale3
    return z


@_jit(4)
def _ppf(p, xmin, xmax, loc, scale):
    T = type(xmin)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    pstar = p * (pmax - pmin) + pmin
    z = _expon._ppf(pstar)
    x = z * scale + loc
    return x


def logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(_cast(x), xmin, xmax, loc, scale)


def pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(_cast(x), xmin, xmax, loc, scale))


def cdf(x, xmin, xmax, loc, scale):
    """
    Return cumulative probability.
    """
    return _cdf(_cast(x), xmin, xmax, loc, scale)


def ppf(p, xmin, xmax, loc, scale):
    """
    Return quantile for given probability.
    """
    return _ppf(_cast(p), xmin, xmax, loc, scale)
