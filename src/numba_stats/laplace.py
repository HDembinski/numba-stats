"""
Laplace distribution.

See Also
--------
scipy.stats.laplace: Scipy equivalent.
"""

import numpy as np
from ._util import _jit, _trans, _generate_wrappers, _prange, _rvs_jit, _seed

_doc_par = """
loc : float
    Location of the mode.
scale : float
    Standard deviation.
"""


@_jit(1, narg=0)
def _cdf1(z):
    return 1.0 - 0.5 * np.exp(-z) if z > 0 else 0.5 * np.exp(z)


@_jit(1, narg=0)
def _ppf1(p):
    return -np.log(2 * (1 - p)) if p > 0.5 else np.log(2 * p)


@_jit(2)
def _logpdf(x, loc, scale):
    z = _trans(x, loc, scale)
    r = np.empty_like(z)
    for i in _prange(len(r)):
        r[i] = np.log(0.25) - np.abs(z[i])
    return r


@_jit(2)
def _pdf(x, loc, scale):
    return np.exp(_logpdf(x, loc, scale))


@_jit(2)
def _cdf(x, loc, scale):
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
    return np.random.laplace(loc, scale, size)


_generate_wrappers(globals())
