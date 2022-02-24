"""
Voigtian distribution.

This is the convolution of a Cauchy distribution with a normal distribution.

There is a closed form for the cdf, but the required hypergeometric function is not
implemented anywhere.

https://en.wikipedia.org/wiki/Voigt_profile
"""

from ._special import voigt_profile as _voigt
from ._util import _vectorize


@_vectorize(4, cache=False)
def pdf(x, gamma, loc, scale):
    """
    Return probability density.
    """
    return _voigt(x - loc, scale, gamma)
