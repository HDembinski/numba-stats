"""
Lognormal distribution.
"""
import numpy as np
from . import norm as _norm
from ._util import _jit, _trans, _generate_wrappers


@_jit(3)
def _logpdf(x, s, loc, scale):
    """
    Return log of probability density.
    """
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        if ri > 0:
            r[i] = -0.5 * np.log(ri) ** 2 / s**2 - np.log(
                s * ri * np.sqrt(2 * np.pi) * scale
            )
        else:
            r[i] = -np.inf
    return r


@_jit(3)
def _pdf(x, s, loc, scale):
    """
    Return probability density.
    """
    return np.exp(_logpdf(x, s, loc, scale))


@_jit(3)
def _cdf(x, s, loc, scale):
    """
    Return cumulative probability.
    """
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        if ri <= 0:
            r[i] = 0.0
        else:
            ri = np.log(ri) / s
            r[i] = _norm._cdf1(ri)
    return r


@_jit(3, cache=False)  # no cache because of norm._ppf
def _ppf(p, s, loc, scale):
    """
    Return quantile for given probability.
    """
    r = np.empty_like(p)
    for i in range(len(p)):
        zi = np.exp(s * _norm._ppf1(p[i]))
        r[i] = scale * zi + loc
    return r


_generate_wrappers(globals())
