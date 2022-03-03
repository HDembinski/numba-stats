"""
Empirical density distribution formed by a Bernstein polynomial.

The Bernstein polynomial basis is better suited to model a probability distribution
than the Chebychev basis, since it is possible to implement the constraint
f(x; p) >= 0 with simple parameter limits p >= 0 (where p is a vector).

The density function and its integral are not normalised. This is not an issue when
the density is used in an extended maximum-likelihood fit.
"""

import numpy as np
import numba as nb


_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.njit(_signatures, cache=True)
def _de_castlejau(z, beta, res):
    # De Casteljau algorithm, numerically stable
    n = len(beta)
    if n == 0:
        res[:] = np.nan
    else:
        betai = np.empty_like(beta)
        for iz, zi in enumerate(z):
            azi = 1.0 - zi
            betai[:] = beta
            for j in range(1, n):
                for k in range(n - j):
                    betai[k] = betai[k] * azi + betai[k + 1] * zi
            res[iz] = betai[0]
    return res


_signatures = [
    nb.float32[:](nb.float32[:]),
    nb.float64[:](nb.float64[:]),
]


@nb.njit(_signatures, cache=True)
def _beta_int(beta):
    n = len(beta)
    r = np.zeros(n + 1, dtype=beta.dtype)
    for j in range(1, n + 1):
        for k in range(j):
            r[j] += beta[k]
    r *= 1.0 / n
    return r


@nb.njit(cache=True)
def _prepare_z_beta(x, xmin, xmax, beta):
    z = x - xmin
    z *= 1 / (xmax - xmin)
    return z, beta


@nb.njit(cache=True)
def _density(x, beta, xmin, xmax, res):
    z, beta = _prepare_z_beta(x, xmin, xmax, beta)
    _de_castlejau(z, beta, res)


@nb.njit(cache=True)
def _integral(x, beta, xmin, xmax, res):
    z, beta = _prepare_z_beta(x, xmin, xmax, beta)
    beta = _beta_int(beta) * (xmax - xmin)
    _de_castlejau(z, beta, res)


_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.guvectorize(_signatures, "(),(n),(),()->()", cache=True)
def density(x, beta, xmin, xmax, res):
    _density(x, beta, xmin, xmax, res)


@nb.guvectorize(_signatures, "(),(n),(),()->()", cache=True)
def integral(x, beta, xmin, xmax, res):
    _integral(x, beta, xmin, xmax, res)


@nb.extending.overload(density)
def density_ol(x, beta, xmin, xmax):
    from numba.core.errors import TypingError
    from numba.types import Array, Float

    if not isinstance(x, Array):
        raise TypingError("x must be a Numpy array")
    if not isinstance(beta, Array):
        raise TypingError("beta must be a Numpy array")
    if not isinstance(xmin, Float):
        raise TypingError("xmin must be float")
    if not isinstance(xmax, Float):
        raise TypingError("xmax must be float")

    def wrap(x, beta, xmin, xmax):
        res = np.empty_like(x)
        _density(x, beta, xmin, xmax, res)
        return res

    return wrap


@nb.extending.overload(integral)
def integral_ol(x, beta, xmin, xmax):
    from numba.core.errors import TypingError
    from numba.types import Array, Float

    if not isinstance(x, Array):
        raise TypingError("x must be a Numpy array")
    if not isinstance(beta, Array):
        raise TypingError("beta must be a Numpy array")
    if not isinstance(xmin, Float):
        raise TypingError("xmin must be float")
    if not isinstance(xmax, Float):
        raise TypingError("xmax must be float")

    def wrap(x, beta, xmin, xmax):
        res = np.empty_like(x)
        _integral(x, beta, xmin, xmax, res)
        return res

    return wrap


# for backward compatibility, avoid in new code
scaled_pdf = density
scaled_cdf = integral
