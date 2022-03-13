"""
Exponential distribution.

See Also
--------
scipy.stats.expon: Scipy equivalent.
"""
import numpy as np
from math import expm1 as _expm1, log1p as _log1p
from ._util import _jit, _trans, _generate_wrappers, _prange

_doc_par = """
x: ArrayLike
    Random variate.
loc : float
    Location of the mode.
scale : float
    Standard deviation.
"""


@_jit(-1)
def _cdf1(z):
    return -_expm1(-z)


@_jit(-1)
def _ppf1(p):
    return -_log1p(-p)


@_jit(2)
def _logpdf(x, loc, scale):
    z = _trans(x, loc, scale)
    return -z - np.log(scale)


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


_generate_wrappers(globals())
