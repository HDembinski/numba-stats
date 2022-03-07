"""
Truncated normal distribution.

See Also
--------
scipy.stats.truncnorm: Scipy equivalent.
"""

import numpy as np
from . import norm as _norm
from ._util import _jit, _generate_wrappers

_doc_par = """
x: ArrayLike
    Random variate.
xmin : float
    Lower edge of the distribution.
xmin : float
    Upper edge of the distribution.
loc : float
    Location of the mode.
scale : float
    Width parameter.
"""


@_jit(4)
def _logpdf(x, xmin, xmax, loc, scale):
    scale2 = type(scale)(1) / scale
    z = (x - loc) * scale2
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    scale *= _norm._cdf1(zmax) - _norm._cdf1(zmin)
    for i, zi in enumerate(z):
        if zmin <= zi < zmax:
            z[i] = _norm._logpdf1(zi) - np.log(scale)
        else:
            z[i] = -np.inf
    return z


@_jit(4)
def _pdf(x, xmin, xmax, loc, scale):
    return np.exp(_logpdf(x, xmin, xmax, loc, scale))


@_jit(4)
def _cdf(x, xmin, xmax, loc, scale):
    scale = type(scale)(1) / scale
    r = (x - loc) * scale
    zmin = (xmin - loc) * scale
    zmax = (xmax - loc) * scale
    pmin = _norm._cdf1(zmin)
    pmax = _norm._cdf1(zmax)
    for i, ri in enumerate(r):
        if zmin <= ri < zmax:
            r[i] = (_norm._cdf1(ri) - pmin) / (pmax - pmin)
        elif ri < zmin:
            r[i] = 0.0
        else:
            r[i] = 1.0
    return r


@_jit(4, cache=False)
def _ppf(p, xmin, xmax, loc, scale):
    scale2 = type(scale)(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _norm._cdf1(zmin)
    pmax = _norm._cdf1(zmax)
    r = p * (pmax - pmin) + pmin
    for i, ri in enumerate(r):
        r[i] = scale * _norm._ppf1(ri) + loc
    return r


_generate_wrappers(globals())
