import numba as nb
import numpy as np
from math import gamma, lgamma
from ._special import hyp2f1
from . import norm


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


# @nb.vectorize(_signatures)
# def cdf(x, q, mu, sigma):
#     inv_scale = 1.0 / sigma
#     z = (x - mu) * inv_scale
#     c_q = _compute_cq(q)
#     alpha = 1.0 - q
#     return hyp2f1(0.5, -1.0 / alpha, 1.5, 0.5 * z ** 2 * alpha) / c_q + 0.5


@nb.vectorize(_signatures)
def cdf(x, q, mu, sigma):
    if q < 1 or q > 2:
        raise ValueError("q < 1 or q >= 3 are not supported")

    if q == 1:
        return norm.cdf(x, mu, sigma)

    # 1 / (2 sigma^2) = 1 / (3 - q)
    # sqrt((3 - q) / 2) = sigma

    sigma *= np.sqrt(0.5 * (3.0 - q))
    inv_scale = 1.0 / sigma
    z = (x - mu) * inv_scale
    nu = (3 - q) / (q - 1)

    return 0.5 + z * gamma(0.5 * (nu + 1)) / np.sqrt(nu * np.pi) / gamma(
        0.5 * nu
    ) * hyp2f1(0.5, 0.5 * (nu + 1), 1.5, -(z ** 2) / nu)