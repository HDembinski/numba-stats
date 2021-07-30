import numba as nb
from ._special import gammaincc


_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def cdf(x, mu):
    """
    Evaluate cumulative distribution function of continuous Poisson distribution.
    """
    return gammaincc(x + 1, mu)


# The pdf, d cdf(x, mu)/ dx, cannot be expressed in tabulated functions:
#
# d G(x, mu)/d x = ln(mu) G(x, mu) + mu T(3, x, mu)
#
# where G(x, mu) is the upper incomplete gamma function and T(m, s, x) is a special case
# of the Meijer G-function,
# see https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives
#
# There is a Meijer G-function implemented in mpmath, but I don't know how to use it.
