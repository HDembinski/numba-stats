"""
Generalised ARGUS distribution.

The ARGUS distribution is named after the particle physics experiment ARGUS and it describes the reconstructed invariant mass of a decayed particle candidate in continuum background.
It is motivated from experimental observation. Here we have the generalised version of the ARGUS distribution that can be used to describe a more peaking like distribtion. p = 0.5 gives the normal ARGUS distribution.

https://en.wikipedia.org/wiki/ARGUS_distribution

See Also
--------
scipy.stats.argus: Scipy equivalent.
"""

from math import gamma

import numpy as np

from ._special import gammaincc
from ._util import _generate_wrappers, _jit, _prange

_doc_par = """
x : Array-like
    Random variate, between 0 and c.
chi : float
    Must be larger than 0 and represents the cutoff.
c : float
    Must be larger than 0 and represents the curvature.
p : float
    Must be larger than -1 and represents the power.
"""


@_jit(3, cache=False)
def _logpdf(x, chi, c, p):
    T = type(p)

    y = T(1) - (x ** T(2)) / (c ** T(2))
    z = (
        -T(0.5) * chi * chi * y
        + p * np.log(y)
        - p * np.log(T(2))
        + (T(2) * p + T(2)) * np.log(chi)
        - T(2.0) * np.log(c)
        + np.log(x)
        - np.log(
            T(gamma(p + T(1)))
            - T(gammaincc(p + T(1), T(0.5) * chi ** T(2)) * T(gamma(p + T(1)))),
        )
    )
    return z


@_jit(3, cache=False)
def _pdf(x, chi, c, p):
    return np.exp(_logpdf(x, chi, c, p))


@_jit(3, cache=False)
def _cdf(x, chi, c, p):
    T = type(p)
    r = np.empty_like(x)
    for i in _prange(len(x)):
        r[i] = (
            (
                gammaincc(
                    p + T(1.0), T(0.5) * chi ** T(2) * (T(1) - x[i] ** T(2) / c ** T(2))
                )
                * gamma(p + T(1))
                - T(gammaincc(p + T(1.0), T(0.5) * chi ** T(2)) * gamma(p + T(1)))
            )
        ) / (
            T(gamma(p + T(1)))
            - T(gammaincc(p + T(1), T(0.5) * chi ** T(2)) * gamma(p + T(1)))
        )
    return r


_generate_wrappers(globals())
