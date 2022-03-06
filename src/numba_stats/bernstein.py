"""
Empirical density distribution formed by a Bernstein polynomial.

The Bernstein polynomial basis is better suited to model a probability distribution
than the Chebychev basis, since it is possible to implement the constraint
f(x; p) >= 0 with simple parameter limits p >= 0 (where p is a vector).

The density function and its integral are not normalised. Normalising would create
several issues while providing no practical benefit. Use this function in an extended
maximum-likelihood fit and do not multiply it with a scale. The scale is implicit in the
parameter vector beta. If you really need to know the scale after the fit (usually you
do not), use :func:`integral` to compute it.
"""

import numpy as np
from ._util import _jit, _Floats


@_jit([T[:](T[:], T[:]) for T in _Floats])
def _de_castlejau(z, beta):
    # De Casteljau algorithm, numerically stable
    n = len(beta)
    res = np.empty_like(z)
    if n == 0:
        res[:] = np.nan
    else:
        betai = np.empty_like(beta)
        for i, zi in enumerate(z):
            azi = 1.0 - zi
            betai[:] = beta
            for j in range(1, n):
                for k in range(n - j):
                    betai[k] = betai[k] * azi + betai[k + 1] * zi
            res[i] = betai[0]
    return res


@_jit(0)
def _beta_int(beta):
    n = len(beta)
    r = np.zeros(n + 1, dtype=beta.dtype)
    for j in range(1, n + 1):
        for k in range(j):
            r[j] += beta[k]
    r *= 1.0 / n
    return r


@_jit(2)
def _trans(x, xmin, xmax):
    scale = type(xmin)(1) / (xmax - xmin)
    return (x - xmin) * scale


@_jit([T[:](T[:], T[:], T, T) for T in _Floats])
def density(x, beta, xmin, xmax):
    """
    Return density described by a Bernstein polynomial.

    The function is non-negative, if each element of the parameter vector beta is non-
    negative, see module description.

    This function is by design equivalent to
    ``scipy.interpolate.BPoly(np.array(beta)[:, np.newaxis], [x[0], x[-1]])(x)``.

    Parameters
    ----------
    x : ArrayLike
        Values where the density is evaluated.
    beta : ArrayLike
        Vector of parameters (1D).
    xmin : float
        Lower edge of the domain of x.
    xmax : float
        Upper edge of the domain of x.

    Returns
    -------
    ndarray
        Function values.
    """
    z = _trans(x, xmin, xmax)
    return _de_castlejau(z, beta)


@_jit([T[:](T[:], T[:], T, T) for T in _Floats], cache=True)
def integral(x, beta, xmin, xmax):
    """
    Return integral of a Bernstein polynomial from xmin to x.

    Parameters
    ----------
    x : ArrayLike
        Values up to which the integral is computed, starting from xmin.
    beta : ArrayLike
        Vector of parameters (1D).
    xmin : float
        Lower edge of the domain of x.
    xmax : float
        Upper edge of the domain of x.

    Returns
    -------
    ndarray
        Integral values.
    """
    z = _trans(x, xmin, xmax)
    beta = _beta_int(beta) * (xmax - xmin)
    return _de_castlejau(z, beta)


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
