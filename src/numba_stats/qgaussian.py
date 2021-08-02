import numba as nb
import numpy as np
from math import lgamma
from . import norm, t


@nb.njit
def _qexp(x, q):
    if q == 1:
        return np.exp(x)
    alpha = 1.0 - q
    arg = 1.0 + alpha * x
    if arg <= 0:
        return 0
    le = np.log(arg) / alpha
    return np.exp(le)


@nb.njit
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
            * np.exp(lgamma(1.0 / alpha) - lgamma(0.5 * (3.0 - q) / alpha))
        )
    if q < 3:
        return (
            const
            / np.sqrt(-alpha)
            * np.exp(lgamma(-0.5 * (3.0 - q) / alpha) - lgamma(-1.0 / alpha))
        )
    return np.nan


@nb.njit
def _df_sigma(q, sigma):
    # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    # relation to Student's t-distribution

    # 1/(2 sigma^2) = 1 / (3 - q)
    # 2 sigma^2 = 3 - q
    # sigma = sqrt((3 - q)/2)

    df = (3 - q) / (q - 1)
    sigma /= np.sqrt(0.5 * (3 - q))

    return df, sigma


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def pdf(x, q, mu, sigma):
    inv_scale = 1.0 / sigma
    z = (x - mu) * inv_scale
    c_q = _compute_cq(q)
    inv_scale /= c_q
    # beta = 1/2 for equivalence with normal distribution for q = 1
    if q == 1.0:
        return np.exp(-0.5 * z ** 2) * inv_scale
    return _qexp(-0.5 * z ** 2, q) * inv_scale


@nb.vectorize(_signatures)
def cdf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return norm.cdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return t.cdf(x, df, mu, sigma)


@nb.vectorize(_signatures)
def ppf(x, q, mu, sigma):
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return norm.ppf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return t.ppf(x, df, mu, sigma)
