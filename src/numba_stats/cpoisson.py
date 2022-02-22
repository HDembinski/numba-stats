"""
Continuous Poisson distribution.
"""
from ._special import gammaincc as _gammaincc
from ._util import _vectorize


@_vectorize(2, cache=False)
def cdf(x, mu):
    """
    Return cumulative probability.
    """
    return _gammaincc(x + 1, mu)


# The pdf, d cdf(x, mu)/ dx, cannot be expressed in tabulated functions:
#
# d G(x, mu)/d x = ln(mu) G(x, mu) + mu T(3, x, mu)
#
# where G(x, mu) is the upper incomplete gamma function and T(m, s, x) is a special case
# of the Meijer G-function,
# see https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives
#
# There is a Meijer G-function implemented in mpmath, but I don't know how to use it.
