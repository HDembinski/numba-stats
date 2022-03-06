"""
Truncated normal distribution.
"""

import numpy as np
from . import norm as _norm
from ._util import _jit


@_jit(4)
def logpdf(x, xmin, xmax, loc, scale):
    """
    Return log of probability density.
    """
    scale2 = type(scale)(1) / scale
    z = (x - loc) * scale2
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    scale *= _norm._cdf(zmax) - _norm._cdf(zmin)
    for i, zi in enumerate(z):
        if zmin <= zi < zmax:
            z[i] = _norm._logpdf(zi) - np.log(scale)
        else:
            z[i] = -np.inf
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
    scale = type(scale)(1) / scale
    r = (x - loc) * scale
    zmin = (xmin - loc) * scale
    zmax = (xmax - loc) * scale
    pmin = _norm._cdf(zmin)
    pmax = _norm._cdf(zmax)
    for i, ri in enumerate(r):
        if zmin <= ri < zmax:
            r[i] = (_norm._cdf(ri) - pmin) / (pmax - pmin)
        elif ri < zmin:
            r[i] = 0.0
        else:
            r[i] = 1.0
    return r


@_jit(4, cache=False)
def ppf(p, xmin, xmax, loc, scale):
    """
    Return quantile for given probability.
    """
    scale2 = type(scale)(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _norm._cdf(zmin)
    pmax = _norm._cdf(zmax)
    r = p * (pmax - pmin) + pmin
    for i, ri in enumerate(r):
        r[i] = scale * _norm._ppf(ri) + loc
    return r
