"""
Cruijff distribution.

See For example: https://arxiv.org/abs/1005.4087
"""

from ._util import _jit, _generate_wrappers, _prange
import numpy as np
from scipy.integrate import quad
import numba

_doc_par = """
x : ArrayLike
    Random variate.
mean : float
    mean of the distribution
sigma_left : float
    left width
sigma_right: float
    right width
alpha_left: float
    left tail parameter
alpha_right: float
    right tail parameter
"""


@_jit(-4)
def _logpdf_gen(x, mean, sigma, alpha):
    return -((x - mean) ** 2) / (2 * sigma * sigma + alpha * (x - mean) ** 2)


@_jit(5)
def _logpdf(x, mean, sigma_left, sigma_right, alpha_left, alpha_right):
    r = np.empty_like(x)
    for i in _prange(len(x)):
        if x[i] < mean:
            r[i] = _logpdf_gen(x[i], mean, sigma_left, alpha_left)
        elif x[i] > mean:
            r[i] = _logpdf_gen(x[i], mean, sigma_right, alpha_right)
    return r


@_jit(5)
def _density(x, mean, sigma_left, sigma_right, alpha_left, alpha_right):
    return np.exp(_logpdf(x, mean, sigma_left, sigma_right, alpha_left, alpha_right))


@_jit(-5)
def _norm(mean, sigma_left, sigma_right, alpha_left, alpha_right):
    with numba.objmode(value="float"):
        (value,) = quad(
            lambda x: _density(
                x, mean, sigma_left, sigma_right, alpha_left, alpha_right
            ),
            -np.inf,
            np.inf,
        )
    return value


@_jit(5)
def _pdf(x, mean, sigma_left, sigma_right, alpha_left, alpha_right):
    return _density(x, mean, sigma_left, sigma_right, alpha_left, alpha_right) / _norm(
        mean, sigma_left, sigma_right, alpha_left, alpha_right
    )


_generate_wrappers(globals())
