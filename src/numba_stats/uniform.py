"""
Uniform distribution.

See Also
--------
scipy.stats.uniform: Equivalent in Scipy.
"""

from ._util import _jit, _generate_wrappers, _prange, _rvs_jit, _seed
import numpy as np

_doc_par = """
a : float
    Lower edge of the distribution.
w : float
    Width of the distribution.
"""


@_jit(2)
def _logpdf(x, a, w):
    r = np.empty_like(x)
    for i in _prange(len(x)):
        if a <= x[i] <= a + w:
            r[i] = -np.log(w)
        else:
            r[i] = -np.inf
    return r


@_jit(2)
def _pdf(x, a, w):
    return np.exp(_logpdf(x, a, w))


@_jit(2)
def _cdf(x, a, w):
    r = np.empty_like(x)
    for i in _prange(len(x)):
        if a <= x[i]:
            if x[i] <= a + w:
                r[i] = (x[i] - a) / w
            else:
                r[i] = 1
        else:
            r[i] = 0
    return r


@_jit(2)
def _ppf(p, a, w):
    return w * p + a


@_rvs_jit(2)
def _rvs(a, w, size, random_state):
    _seed(random_state)
    return np.random.uniform(a, a + w, size)


_generate_wrappers(globals())
