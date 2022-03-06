"""
Continuous Poisson distribution.
"""
from ._special import gammaincc as _gammaincc
from ._util import _jit, _wrap
import numpy as np


@_jit(1, cache=False)
def _cdf(x, mu):
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        r[i] = _gammaincc(xi + type(xi)(1), mu)
    return r


def cdf(x, mu):
    """
    Return cumulative probability.
    """
    return _wrap(_cdf)(x, mu)


# The pdf, d cdf(x, mu)/ dx, cannot be expressed in tabulated functions:
#
# d G(x, mu)/d x = ln(mu) G(x, mu) + mu T(3, x, mu)
#
# where G(x, mu) is the upper incomplete gamma function and T(m, s, x) is a special case
# of the Meijer G-function,
# see https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives
#
# There is a Meijer G-function implemented in mpmath, but I don't know how to use it.
