"""
Q-Gaussian distribution.

A generalisation (q-analog) of the normal distribution based on Tsallis entropy. It
can be used an alternative model for the normal distribution to check for model bias.

It is equivalent to Student's t distribution and can be computed from the latter via
a change of variables, which is exploited in this implementation.

https://en.wikipedia.org/wiki/Q-Gaussian_distribution
"""

import numpy as np
import numba as nb
from . import norm as _norm, t as _t
from ._util import _jit, _generate_wrappers

_doc_par = """
x : ArrayLike
    Random variate.
q : float
    Shape parameter between 1 and 3. For q = 1, the qgaussian is a normal distribution,
    for q == 3 it is a cauchy distribution.
loc : float
    Expectation value.
scale : float
    Width parameter.
"""


@nb.njit
def _df_sigma(q, sigma):
    # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    # relation to Student's t-distribution

    # 1/(2 sigma^2) = 1 / (3 - q)
    # 2 sigma^2 = 3 - q
    # sigma = sqrt((3 - q)/2)
    T = type(q)
    df = (T(3) - q) / (q - T(1))
    sigma /= np.sqrt(T(0.5) * (T(3) - q))

    return df, sigma


@_jit(3)
def _logpdf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._logpdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._logpdf(x, df, mu, sigma)


@_jit(3)
def _pdf(x, q, mu, sigma):
    return np.exp(_logpdf(x, q, mu, sigma))


@_jit(3, cache=False)
def _cdf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._cdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._cdf(x, df, mu, sigma)


@_jit(3, cache=False)
def _ppf(p, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._ppf(p, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._ppf(p, df, mu, sigma)


_generate_wrappers(globals())
