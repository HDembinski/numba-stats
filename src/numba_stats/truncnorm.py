"""
Truncated normal distribution.
"""

import numpy as np
from . import norm as _norm
from ._util import _jit, _cast


@_jit(4)
def _logpdf(x, xmin, xmax, loc, scale):
    scale_inv = 1 / scale
    z = (x - loc) * scale_inv
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    scale *= _norm._cdfz(zmax) - _norm._cdfz(zmin)
    for i, zi in enumerate(z):
        if zmin <= zi < zmax:
            z[i] = _norm._logpdfz(zi) - np.log(scale)
        else:
            z[i] = -np.inf
    return z


@_jit(4)
def _pdf(x, xmin, xmax, loc, scale):
    return np.exp(_logpdf(x, xmin, xmax, loc, scale))


@_jit(4)
def _cdf(x, xmin, xmax, loc, scale):
    scale_inv = 1 / scale
    r = (x - loc) * scale_inv
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    pmin = _norm._cdfz(zmin)
    pmax = _norm._cdfz(zmax)
    for i, ri in enumerate(r):
        if zmin <= ri < zmax:
            r[i] = (_norm._cdfz(ri) - pmin) / (pmax - pmin)
        elif ri < zmin:
            r[i] = 0.0
        else:
            r[i] = 1.0
    return r


@_jit(4, cache=False)
def _ppf(p, xmin, xmax, loc, scale):
    scale_inv = 1 / scale
    zmin = (xmin - loc) * scale_inv
    zmax = (xmax - loc) * scale_inv
    pmin = _norm._cdfz(zmin)
    pmax = _norm._cdfz(zmax)
    r = p * (pmax - pmin) + pmin
    for i, ri in enumerate(r):
        r[i] = scale * _norm._ppfz(ri) + loc
    return r


def logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(_cast(x), xmin, xmax, loc, scale)


def pdf(x, xmin, xmax, loc, scale):
    """
    Return probability density.
    """
    return _pdf(_cast(x), xmin, xmax, loc, scale)


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
