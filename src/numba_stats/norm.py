"""
Normal distribution.

See Also
--------
scipy.stats.norm: Scipy equivalent.
"""
import numpy as np
from ._special import ndtri as _ndtri
from ._util import _jit, _trans, _generate_wrappers, _prange, _to_array
from math import erf as _erf

_doc_par = """
x : ArrayLike
    Random variate.
loc : float
    Location of the mode of the distribution.
scale : float
    Standard deviation.
"""


@_jit(-1)
def _logpdf1(z):
    T = type(z)
    return -T(0.5) * (z * z + T(np.log(2 * np.pi)))


@_jit(-1)
def _cdf1(z):
    T = type(z)
    c = T(np.sqrt(0.5))
    return T(0.5) * (T(1.0) + _erf(z * c))


@_jit(-1, cache=False)  # cannot cache because of _ndtri
def _ppf1(p):
    T = type(p)
    return T(_ndtri(p))


@_jit(2)
def _logpdf(x, loc, scale):
    x, shape = _to_array(x)
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i]) - np.log(scale)
    return np.reshape(r, shape)


@_jit(2)
def _pdf(x, loc, scale):
    return np.exp(_logpdf(x, loc, scale))


@_jit(2)
def _cdf(x, loc, scale):
    x, shape = _to_array(x)
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _cdf1(r[i])
    return np.reshape(r, shape)


@_jit(2, cache=False)
def _ppf(p, loc, scale):
    p, shape = _to_array(p)
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i]) + loc
    return np.reshape(r, shape)


_generate_wrappers(globals())
