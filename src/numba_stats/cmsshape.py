"""
CMS-Shape distribution (for lack of a better name).

The distribution consists of an exponential decay suppressed at small values by the
complementary error function. The product is an asymmetric peak with a bell shape on the
left hand side and an exponential tail on the right hand side. This shape is used by the
CMS experiment to model the background in the invariant mass distribution of Z to ll
decay candidates.

Notes
-----
This implementation was modeled after
https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooCMSShape.cc, but heavily
modified. We added an analytical normalization and an analytical integral. It turns out
that the parameters "alpha" and "peak" in the original implementation are redundant.
They have been replaced with a single parameter "loc", which is the approximate center
of the distribution.
"""

from math import erf as _erf, erfc as _erfc
import numpy as np
from ._util import _jit, _generate_wrappers, _prange

_doc_par = """
beta : float
    Steepness of the error function. Must be positive.
gamma : float
    Steepness of the exponential distribution. Must be positive.
loc: float
    Approximate center of the distribution.
"""


@_jit(3)
def _logpdf(x, beta, gamma, loc):
    T = type(beta)
    two = T(2)
    half = T(0.5)
    v = -(x - loc) * beta
    for i in _prange(len(x)):
        v[i] = _erfc(v[i])
    u = (x - loc) * gamma
    T = type(beta)
    log_t = (half * gamma / beta) ** two
    return np.log(v) - u + np.log(half * gamma) - log_t


@_jit(3)
def _pdf(x, beta, gamma, loc):
    return np.exp(_logpdf(x, beta, gamma, loc))


@_jit(3)
def _cdf(x, beta, gamma, loc):
    T = type(beta)
    y = x - loc
    two = T(2)
    half = T(0.5)
    t1 = gamma / (two * beta) + beta * y
    for i in _prange(len(t1)):
        t1[i] = _erf(t1[i])
    t2 = np.exp(-((gamma / (two * beta)) ** 2) - gamma * y)
    t3 = -beta * y
    for i in _prange(len(t1)):
        t3[i] = _erfc(t3[i])
    return half * (t1 - t2 * t3) + half


_generate_wrappers(globals())
