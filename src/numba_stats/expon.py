"""
Exponential distribution.

See Also
--------
scipy.stats.expon: Scipy equivalent.
"""
import numpy as np
from math import expm1 as _expm1, log1p as _log1p
from ._util import _jit, _trans, _generate_wrappers, _prange, _rvs_jit, _seed

_doc_par = """
loc : float
    Location of the mode.
scale : float
    Standard deviation.
"""


@_jit(-1)
def _cdf1(z):
    T = type(z)
    return T(0) if z < 0 else -_expm1(-z)


@_jit(-1)
def _ppf1(p):
    return -_log1p(-p)


@_jit(2)
def _logpdf(x, loc, scale):
    z = _trans(x, loc, scale)
    r = np.empty_like(z)
    for i in _prange(len(r)):
        r[i] = -np.inf if z[i] < 0 else -z[i] - np.log(scale)
    return r


@_jit(2)
def _pdf(x, loc, scale):
    """
    Return probability density.

    Parameters
    ----------
    x: ArrayLike
        Random variate.
    loc : float
        Location of the mode.
    scale : float
        Standard deviation.

    Returns
    -------
    ArrayLike
        Function evaluated at x.
    """
    return np.exp(_logpdf(x, loc, scale))


@_jit(2)
def _cdf(x, loc, scale):
    """
    Return cumulative probability.

    Parameters
    ----------
    x: ArrayLike
        Random variate.
    loc : float
        Location of the mode.
    scale : float
        Standard deviation.

    Returns
    -------
    ArrayLike
        Function evaluated at x.
    """
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _cdf1(z[i])
    return z


@_jit(2)
def _ppf(p, loc, scale):
    z = np.empty_like(p)
    for i in _prange(len(z)):
        z[i] = _ppf1(p[i])
    return scale * z + loc


@_rvs_jit(2)
def _rvs(loc, scale, size, random_state):
    _seed(random_state)
    p = np.random.uniform(0, 1, size)
    return _ppf(p, loc, scale)


_generate_wrappers(globals())
