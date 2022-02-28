"""
Q-Gaussian distribution.

A generalisation (q-analog) of the normal distribution based on Tsallis entropy. It
can be used an alternative model for the normal distribution to check for model bias.

It is equivalent to Student's t distribution and can be computed from the latter via
a change of variables, which is exploited in this implementation.

https://en.wikipedia.org/wiki/Q-Gaussian_distribution
"""

import numpy as np
from math import lgamma as _lgamma
from . import norm as _norm, t as _t
from ._util import _jit, _vectorize


@_jit
def _qexp(x, q):
    if q == 1:
        return np.exp(x)
    alpha = 1.0 - q
    arg = 1.0 + alpha * x
    if arg <= 0:
        return 0
    le = np.log(arg) / alpha
    return np.exp(le)


@_jit
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


@_jit
def _df_sigma(q, sigma):
    # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    # relation to Student's t-distribution

    # 1/(2 sigma^2) = 1 / (3 - q)
    # 2 sigma^2 = 3 - q
    # sigma = sqrt((3 - q)/2)

    df = (3 - q) / (q - 1)
    sigma /= np.sqrt(0.5 * (3 - q))

    return df, sigma


@_vectorize(4)
def pdf(x, q, mu, sigma):
    inv_scale = 1.0 / sigma
    z = (x - mu) * inv_scale
    c_q = _compute_cq(q)
    inv_scale /= c_q
    # beta = 1/2 for equivalence with normal distribution for q = 1
    if q == 1.0:
        return np.exp(-0.5 * z**2) * inv_scale
    return _qexp(-0.5 * z**2, q) * inv_scale


@_vectorize(4, cache=False)
def cdf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm.cdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t.cdf(x, df, mu, sigma)


@_vectorize(4, cache=False)
def ppf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm.ppf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t.ppf(x, df, mu, sigma)
