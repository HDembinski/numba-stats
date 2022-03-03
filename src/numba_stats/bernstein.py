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


_signatures = [
    nb.float32[:](nb.float32[:], nb.float32[:], nb.float32, nb.float32),
    nb.float64[:](nb.float64[:], nb.float64[:], nb.float64, nb.float64),
]


@nb.njit(_signatures, cache=True)
def _density(x, beta, xmin, xmax):
    z, beta = _prepare_z_beta(x, xmin, xmax, beta)
    res = np.empty_like(x)
    _de_castlejau(z, beta, res)
    return res


@nb.njit(_signatures, cache=True)
def _integral(x, beta, xmin, xmax):
    z, beta = _prepare_z_beta(x, xmin, xmax, beta)
    beta = _beta_int(beta) * (xmax - xmin)
    res = np.empty_like(x)
    _de_castlejau(z, beta, res)
    return res


def _normalize(x):
    x = np.atleast_1d(x)
    if x.dtype.kind != "f":
        return x.astype(float)
    return x


def density(x, beta, xmin, xmax):
    r = _density(_normalize(x), _normalize(beta), xmin, xmax)
    if np.ndim(x) == 0:
        return np.squeeze(r)
    return r


def integral(x, beta, xmin, xmax):
    r = _integral(_normalize(x), _normalize(beta), xmin, xmax)
    if np.ndim(x) == 0:
        return np.squeeze(r)
    return r


@nb.extending.overload(density)
def _density_ol(x, beta, xmin, xmax):
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

    return _density.__wrapped__


@nb.extending.overload(integral)
def _integral_ol(x, beta, xmin, xmax):
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

    return _integral.__wrapped__


def __getattr__(key):
    # Temporary hack to maintain backward compatibility
    import warnings
    from numpy import VisibleDeprecationWarning

    if key in ("scaled_pdf", "scaled_cdf"):
        r = {"scaled_pdf": "density", "scaled_cdf": "integral"}
        warnings.warn(
            f"bernstein.{key} is deprecated and will be removed in a future release, "
            f"use bernstein.{r[key]} instead",
            VisibleDeprecationWarning,
            1,
        )
        return globals()[r[key]]
    raise AttributeError
