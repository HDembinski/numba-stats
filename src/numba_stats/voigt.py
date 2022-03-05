"""
Voigtian distribution.

This is the convolution of a Cauchy distribution with a normal distribution.

There is a closed form for the cdf, but the required hypergeometric function is not
implemented anywhere.

https://en.wikipedia.org/wiki/Voigt_profile
"""

from ._special import voigt_profile as _voigt
from ._util import _jit
import numpy as np


@_jit(3, cache=False)
def _pdf(x, gamma, loc, scale):
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        r[i] = _voigt(xi - loc, scale, gamma)
    return r


def pdf(x, gamma, loc, scale):
    """
    Return probability density.
    """
    return _pdf(x, gamma, loc, scale)
