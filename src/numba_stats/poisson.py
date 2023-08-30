"""
Poisson distribution.

See Also
--------
scipy.stats.poisson: Scipy equivalent.
"""

import numpy as np
from ._special import gammaincc as _gammaincc
from math import lgamma as _lgamma

# from ._util import _generate_wrappers, _prange
import numba as nb

_doc_par = """
x : ArrayLike
    Random variate.
mu : float
    Expected value.
"""

signatures = [
    nb.float64(nb.int64, nb.float64),
    nb.float64(nb.uint64, nb.float64),
    nb.float64(nb.int32, nb.float64),
    nb.float64(nb.uint32, nb.float64),
    nb.float64(nb.float64, nb.float64),
    nb.float32(nb.float32, nb.float32),
    nb.float32(nb.int32, nb.float32),
    nb.float32(nb.uint32, nb.float32),
]


@nb.vectorize(signatures, nopython=True, cache=True)
def logpmf(k, mu):
    """Poisson logpmf."""
    T = type(mu)
    if mu == 0:
        return T(0.0 if k == 0 else -np.inf)
    else:
        return T(k * np.log(mu) - _lgamma(k + T(1)) - mu)


@nb.vectorize(signatures, nopython=True, cache=True)
def pmf(k, mu):
    """Poisson pmf."""
    return np.exp(logpmf(k, mu))


# cannot be cached due to usage of cython function
@nb.vectorize(signatures, nopython=True)
def cdf(k, mu):
    """Poisson cdf."""
    T = type(mu)
    return _gammaincc(k + T(1), mu)


# _generate_wrappers(globals())
