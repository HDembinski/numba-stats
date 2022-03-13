"""
Lognormal distribution.

See Also
--------
scipy.stats.lognorm: Scipy equivalent.
"""
import numpy as np
from . import norm as _norm
from ._util import _jit, _trans, _generate_wrappers, _prange

_doc_par = """
x : ArrayLike
    Random variate.
s : float
    Standard deviation of the corresponding normal distribution of exp(x).
loc : float
    Shift of the distribution.
scale : float
    Equal to exp(mu) of the corresponding normal distribution of exp(x).
"""


@_jit(3)
def _logpdf(x, s, loc, scale):
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        if r[i] > 0:
            r[i] = -0.5 * np.log(r[i]) ** 2 / s**2 - np.log(
                s * r[i] * np.sqrt(2 * np.pi) * scale
            )
        else:
            r[i] = -np.inf
    return r


@_jit(3)
def _pdf(x, s, loc, scale):
    return np.exp(_logpdf(x, s, loc, scale))


@_jit(3)
def _cdf(x, s, loc, scale):
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        if r[i] <= 0:
            r[i] = 0.0
        else:
            z = np.log(r[i]) / s
            r[i] = _norm._cdf1(z)
    return r


@_jit(3, cache=False)  # no cache because of norm._ppf
def _ppf(p, s, loc, scale):
    r = np.empty_like(p)
    for i in _prange(len(p)):
        r[i] = np.exp(s * _norm._ppf1(p[i]))
    return scale * r + loc


_generate_wrappers(globals())
