"""
Tsallis-Hagedorn distribution.

A generalisation (q-analog) of the exponential distribution based on Tsallis entropy. It
approximately describes the pT distribution charged particles produced in high-energy
minimum bias particle collisions.
"""

import numpy as np
from ._util import _jit, _cast


@_jit(3)
def _pdf(x, m, t, n):
    # Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    if n <= 2:
        raise ValueError("n > 2 is required")

    T = type(m)
    mt = np.sqrt(m * m + x * x)
    nt = n * t
    c = (n - T(1)) * (n - T(2)) / (nt * (nt + (n - T(2)) * m))
    return c * x * (T(1) + (mt - m) / nt) ** -n


@_jit(3)
def _cdf(x, m, t, n):
    # Formula computed from tsallis_pdf with Sympy, then simplified by hand
    if n <= 2:
        raise ValueError("n > 2 is required")

    T = type(m)
    mt = np.sqrt(m * m + x * x)
    nt = n * t
    return (
        ((mt - m) / nt + T(1)) ** (T(1) - n)
        * (m + mt - n * (mt + t))
        / (m * (n - T(2)) + nt)
    )


def pdf(x, m, t, n):
    """
    Return probability density.
    """
    return _pdf(_cast(x), m, t, n)


def cdf(x, m, t, n):
    """
    Return cumulative probability.
    """
    return _cdf(_cast(x), m, t, n)
