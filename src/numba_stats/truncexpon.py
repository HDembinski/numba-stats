"""
Truncated exponential distribution.

See Also
--------
scipy.stats.truncexpon: Scipy equivalent.
"""
import numpy as np
from ._util import _jit, _trans, _generate_wrappers, _prange
from . import expon as _expon

_doc_par = """
x: ArrayLike
    Random variate.
xmin : float
    Lower edge of the distribution.
xmax : float
    Upper edge of the distribution.
loc : float
    Location of the mode.
scale : float
    Width parameter.
"""


@_jit(4)
def _logpdf(x, xmin, xmax, loc, scale):
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    c = np.log(scale * (_expon._cdf1(zmax) - _expon._cdf1(zmin)))
    for i in _prange(len(z)):
        if zmin <= z[i] < zmax:
            z[i] = -z[i] - c
        else:
            z[i] = -np.inf
    return z


@_jit(4)
def _pdf(x, xmin, xmax, loc, scale):
    return np.exp(_logpdf(x, xmin, xmax, loc, scale))


@_jit(4)
def _cdf(x, xmin, xmax, loc, scale):
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    scale3 = T(1) / (pmax - pmin)
    for i in _prange(len(z)):
        if zmin <= z[i]:
            if z[i] < zmax:
                z[i] = (_expon._cdf1(z[i]) - pmin) * scale3
            else:
                z[i] = 1
        else:
            z[i] = 0
    return z


@_jit(4)
def _ppf(p, xmin, xmax, loc, scale):
    zmin = (xmin - loc) / scale
    zmax = (xmax - loc) / scale
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    z = p * (pmax - pmin) + pmin
    for i in _prange(len(z)):
        z[i] = _expon._ppf1(z[i])
    return z * scale + loc


_generate_wrappers(globals())
