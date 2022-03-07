"""
Truncated exponential distribution.
"""
import numpy as np
from ._util import _jit, _trans, _generate_wrappers
from . import expon as _expon


@_jit(4)
def _logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
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
            z[i] = -zi - c
    return z


@_jit(4)
def _pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, xmin, xmax, loc, scale))


@_jit(4)
def _cdf(x, xmin, xmax, loc, scale):
    """
    Return cumulative probability.
    """
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    scale3 = T(1) / (pmax - pmin)
    for i, zi in enumerate(z):
        if zmin <= zi:
            if zi < zmax:
                z[i] = (_expon._cdf1(zi) - pmin) * scale3
            else:
                z[i] = 1
        else:
            z[i] = 0
    return z


@_jit(4)
def _ppf(p, xmin, xmax, loc, scale):
    """
    Return quantile for given probability.
    """
    T = type(xmin)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    z = p * (pmax - pmin) + pmin
    for i, zi in enumerate(z):
        z[i] = _expon._ppf1(zi)
    return z * scale + loc


_generate_wrappers(globals())
