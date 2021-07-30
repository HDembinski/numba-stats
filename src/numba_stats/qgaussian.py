import numba as nb
import numpy as np
from math import gamma


@nb.njit
def _qexp(x, q):
    if q == 1:
        return np.exp(x)
    alpha = 1.0 - q
    le = np.log(1.0 + alpha * x) / alpha
    return np.exp(le)


@nb.njit
def _compute_cq(q):
    sqrt_pi = np.sqrt(np.pi)
    alpha = 1.0 - q
    if q == 1:
        return sqrt_pi
    if q < 1:
        return (
            2.0
            * sqrt_pi
            * gamma(1.0 / alpha)
            / ((3.0 - q) * np.sqrt(alpha) * gamma(0.5 * (3.0 - q) / alpha))
        )
    if q < 3:
        return (
            sqrt_pi
            * gamma(-0.5 * (3.0 - q) / alpha)
            / (np.sqrt(-alpha) * gamma(-1.0 / alpha))
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
    inv_scale /= c_q * np.sqrt(2)
    # beta = 1/2 for equivalence with normal dist. for q = 1
    return _qexp(-0.5 * (z ** 2), q) * inv_scale


# from sympy import *
# from matplotlib import pyplot as plt
#
# x = np.linspace(-5, 5)
#
# plt.plot(x, pdf(x, 1, 0, 1))
