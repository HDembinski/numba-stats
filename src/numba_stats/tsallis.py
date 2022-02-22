import numpy as np
from ._util import _vectorize


@_vectorize(4)
def pdf(x, m, t, n):
    # Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    assert n > 2

    mt = np.sqrt(m ** 2 + x ** 2)
    nt = n * t
    c = (n - 1) * (n - 2) / (nt * (nt + (n - 2) * m))
    return c * x * (1 + (mt - m) / nt) ** -n


@_vectorize(4)
def cdf(x, m, t, n):
    """
    Return cumulative probability of Tsallis distribution.
    """
    # Formula computed from tsallis_pdf with Sympy, then simplified by hand
    assert n > 2

    mt = np.sqrt(m ** 2 + x ** 2)
    nt = n * t
    return ((mt - m) / nt + 1) ** (1 - n) * (m + mt - n * (mt + t)) / (m * (n - 2) + nt)
