"""Port of cephes ``polevl.c``.

See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
"""
import numba as nb
from .._util import _Floats, _readonly_carray, _prange


@nb.njit(
    [T(T, _readonly_carray(T)) for T in _Floats],
    cache=True,
    inline="never",
    error_model="numpy",
)
def polevl(x, coefs):
    """
    Evaluate polynomial using Horner's method.

    The degree N of the polynomial is inferred by the **coefs** array.
    In this implementation, though, coefficients are stored in increasing degree order:
    p(x) = c_0 + c_1 * x + ... + c_N * x^N
    where **coefs** is an array with N + 1 length such that
    coefs[0] = c_0, ..., coefs[N] = c_N
    """
    len_coefs = len(coefs)
    ans = coefs[-1]
    try:
        for i in _prange(2, len_coefs + 1):
            ans = ans * x + coefs[-i]
    except Exception:
        # The exception to be catched should be OverflowError,
        # but numba rejects it as the implementation raised a specific error:
        # UnsupportedError: Exception matching is limited to <class 'Exception'>
        pass
    return ans
