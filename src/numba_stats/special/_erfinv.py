import numba as nb
import numpy as np
from numpy import inf, nan
from ._ndtri import ndtri
from .._util import _Floats, _trans


@nb.njit([T(T) for T in _Floats], cache=True, inline="never", error_model="numpy")
def erfinv(z):
    """
    Calculate the inverse error function at point ``z``.

    This is a direct port of the SciPy ``erfinv`` function, originally
    written in C.

    Parameters
    ----------
    z : float

    Returns
    -------
    float

    References
    ----------
    + https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    + http://functions.wolfram.com/GammaBetaErf/InverseErf/

    Examples
    --------
    >>> import math
    >>> round(erfinv(0.1), 12)
    0.088855990494
    >>> round(erfinv(0.5), 12)
    0.476936276204
    >>> round(erfinv(-0.5), 12)
    -0.476936276204
    >>> round(erfinv(0.95), 12)
    1.38590382435
    >>> round(math.erf(erfinv(0.3)), 3)
    0.3
    >>> round(erfinv(math.erf(0.5)), 3)
    0.5
    >>> erfinv(0)
    0
    >>> erfinv(1)
    inf
    >>> erfinv(-1)
    -inf
    """
    T = type(z)
    if np.abs(z) > T(1):
        # If z < -1 or z > 1, we return NaN.
        # TBD: Should the function raise ValueError, signalling explicitly that
        # `z` must be between -1 and 1 inclusive?
        return nan

    # Shortcut special cases
    if z == T(0):
        return T(0)
    if z == T(1):
        return inf
    if z == T(-1):
        return -inf

    # otherwise calculate things.
    inv_sqrt_2 = T(1) / np.sqrt(T(2))
    return ndtri(_trans(z, T(-1), T(2))) * inv_sqrt_2
