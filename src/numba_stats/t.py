"""
Student's t distribution.

See Also
--------
scipy.stats.t: Scipy equivalent.
"""
import numpy as np
from ._special import stdtr as _stdtr, stdtrit as _stdtrit
from ._util import _jit, _trans, _generate_wrappers, _prange
from math import lgamma as _lgamma

_doc_par = """
x: ArrayLike
    Random variate.
df : float
    Degrees of freedom.
loc : float
    Location of the mode.
scale : float
    Width parameter.
"""


@_jit(3, cache=False)
def _logpdf(x, df, loc, scale):
    T = type(df)
    z = _trans(x, loc, scale)
    k = T(0.5) * (df + T(1))
    c = _lgamma(k) - _lgamma(T(0.5) * df)
    c -= T(0.5) * np.log(df * T(np.pi))
    c -= np.log(scale)
    for i in _prange(len(z)):
        z[i] = -k * np.log(T(1) + (z[i] * z[i]) / df) + c
    return z


@_jit(3, cache=False)
def _pdf(x, df, loc, scale):
    return np.exp(_logpdf(x, df, loc, scale))


@_jit(3, cache=False)
def _cdf(x, df, loc, scale):
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _stdtr(df, z[i])
    return z


@_jit(3, cache=False)
def _ppf(p, df, loc, scale):
    T = type(df)
    r = np.empty_like(p)
    for i in _prange(len(p)):
        if p[i] == 0:
            r[i] = -T(np.inf)
        elif p[i] == 1:
            r[i] = T(np.inf)
        else:
            r[i] = _stdtrit(df, p[i])
    return scale * r + loc


_generate_wrappers(globals())
