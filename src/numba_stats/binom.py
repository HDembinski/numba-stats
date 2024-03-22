"""
Binomial distribution.

See Also
--------
scipy.stats.binom: Scipy equivalent.
"""

import numpy as np
from ._special import gammaincc as _gammaincc
from ._special import xlogy as _xlogy
from ._special import xlog1py as _xlog1py
from math import lgamma as _lgamma
from ._util import _jit, _generate_wrappers, _prange, _seed
import numba as nb

_doc_par = """
k : int
    number of successes.
n : int
    number of trails.
p : float
    success probability for each trail.
"""


@_jit(2, cache=False)
def _logpmf(k, n, p):
    T = type(n)
    r = np.empty(len(k), T)
    for i in _prange(len(r)):
        combiln = (_lgamma(n + T(1)) - (_lgamma(k[i] + T(1)) + _lgamma(n-k[i] + T(1))))
        r[i] = combiln + _xlogy(k[i], p) + _xlog1py(n-k[i], -p)
    return r

@_jit(2, cache=False)
def _pmf(k, n, p):
    return np.exp(_logpmf(k, n, p))


@_jit(2, cache=False)
def _cdf(k, n, p):
    T = type(n)
    r = np.empty(len(k), T)
    for i in _prange(len(r)):
        r[i] = np.sum(_pmf(np.arange(0,k[i]+1),n,p))
    return r


@nb.njit(
    nb.int64[:](nb.uint64, nb.float32, nb.uint64, nb.optional(nb.uint64)),
    cache=True,
    inline="always",
    error_model="numpy",
)
def _rvs(n, p, size, random_state):
    _seed(random_state)
    return np.random.binomial(n, p, size=size)


_generate_wrappers(globals())
