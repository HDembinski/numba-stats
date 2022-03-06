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
from math import lgamma as _lgamma
from . import norm as _norm, t as _t
from ._util import _jit, _wrap


@_jit(-2)
def _qexp(x, q):
    if q == 1:
        return np.exp(x)
    alpha = 1.0 - q
    arg = 1.0 + alpha * x
    if arg <= 0:
        return 0
    le = np.log(arg) / alpha
    return np.exp(le)


@_jit(-1)
def _compute_cq(q):
    # beta = 1/2 for equivalence with normal distribution for q = 1
    const = np.sqrt(2 * np.pi)
    alpha = 1.0 - q
    if q == 1:
        return const
    if q < 1:
        return (
            2.0
            * const
            / ((3.0 - q) * np.sqrt(alpha))
            * np.exp(_lgamma(1.0 / alpha) - _lgamma(0.5 * (3.0 - q) / alpha))
        )
    if q < 3:
        return (
            const
            / np.sqrt(-alpha)
            * np.exp(_lgamma(-0.5 * (3.0 - q) / alpha) - _lgamma(-1.0 / alpha))
        )
    return np.nan


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
def _pdf(x, q, mu, sigma):
    T = type(q)
    scale2 = T(1) / sigma
    z = (x - mu) * scale2
    c_q = _compute_cq(q)
    scale2 /= c_q
    # beta = 1/2 for equivalence with normal distribution for q = 1
    if q == 1:
        for i, zi in enumerate(z):
            z[i] = np.exp(-T(0.5) * zi * zi) * scale2
    else:
        for i, zi in enumerate(z):
            z[i] = _qexp(-T(0.5) * zi * zi, q) * scale2
    return z


@_jit(3, cache=False)
def _cdf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._cdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._cdf(x, df, mu, sigma)


@_jit(3, cache=False)
def _ppf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._ppf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._ppf(x, df, mu, sigma)


def pdf(x, q, mu, sigma):
    return _wrap(_pdf)(x, q, mu, sigma)


def cdf(x, q, mu, sigma):
    return _wrap(_cdf)(x, q, mu, sigma)


def ppf(p, q, mu, sigma):
    return _wrap(_ppf)(p, q, mu, sigma)
