"""
Truncated exponential distribution.
"""
import numpy as np
from ._util import _jit, _trans
from . import expon as _expon


@_jit(4)
def logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    c = np.log(scale * (_expon._cdf(zmax) - _expon._cdf(zmin)))
    for i, zi in enumerate(z):
        if zi < zmin:
            z[i] = -T(np.inf)
        elif zi > zmax:
            z[i] = -T(np.inf)
        else:
            z[i] = -zi - c
    return z


@_jit(4)
def pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density.
    """
    return np.exp(logpdf(x, xmin, xmax, loc, scale))


@_jit(4)
def cdf(x, xmin, xmax, loc, scale):
    """
    Return cumulative probability.
    """
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf(zmin)
    pmax = _expon._cdf(zmax)
    scale3 = T(1) / (pmax - pmin)
    for i, zi in enumerate(z):
        if zmin <= zi:
            if zi < zmax:
                z[i] = (_expon._cdf(zi) - pmin) * scale3
            else:
                z[i] = 1
        else:
            z[i] = 0
    return z


@_jit(4)
def ppf(p, xmin, xmax, loc, scale):
    """
    Return quantile for given probability.
    """
    T = type(xmin)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf(zmin)
    pmax = _expon._cdf(zmax)
    z = p * (pmax - pmin) + pmin
    for i, zi in enumerate(z):
        z[i] = _expon._ppf(zi)
    return z * scale + loc
