"""
Tsallis-Hagedorn distribution.

A generalisation (q-analog) of the exponential distribution based on Tsallis entropy. It
approximately describes the pT distribution charged particles produced in high-energy
minimum bias particle collisions.
"""

import numpy as np
from ._util import _vectorize


@_vectorize(4)
def pdf(x, m, t, n):
    """
    Return probability density.
    """
    # Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    if n <= 2:
        raise ValueError("n > 2 is required")

    mt = np.sqrt(m**2 + x**2)
    nt = n * t
    c = (n - 1) * (n - 2) / (nt * (nt + (n - 2) * m))
    return c * x * (1 + (mt - m) / nt) ** -n


@_vectorize(4)
def cdf(x, m, t, n):
    """
    Return cumulative probability.
    """
    # Formula computed from tsallis_pdf with Sympy, then simplified by hand
    if n <= 2:
        raise ValueError("n > 2 is required")

    mt = np.sqrt(m**2 + x**2)
    nt = n * t
    return ((mt - m) / nt + 1) ** (1 - n) * (m + mt - n * (mt + t)) / (m * (n - 2) + nt)
