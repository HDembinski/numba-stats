"""
Lognormal distribution.
"""
import numpy as np
from .norm import _ppfz
from ._util import _jit, _cast, _trans, _type_check
from math import erf as _erf
from numba.extending import overload as _overload


@_jit(3)
def _logpdf(x, s, loc, scale):
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
    return np.exp(_logpdf(x, s, loc, scale))


@_jit(3)
def _cdf(x, s, loc, scale):
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        if ri <= 0:
            r[i] = 0.0
        else:
            ri = np.log(ri) / s
            r[i] = 0.5 * (1.0 + _erf(ri * np.sqrt(0.5)))
    return r


@_jit(3, cache=False)  # no cache because of _ppfz
def _ppf(p, s, loc, scale):
    r = np.empty_like(p)
    for i in range(len(p)):
        zi = np.exp(s * _ppfz(p[i]))
        r[i] = scale * zi + loc
    return r


def logpdf(x, s, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(_cast(x), s, loc, scale)


def pdf(x, s, loc, scale):
    """
    Return probability density.
    """
    return _pdf(_cast(x), s, loc, scale)


def cdf(x, s, loc, scale):
    """
    Return cumulative probability.
    """
    return _cdf(_cast(x), s, loc, scale)


def ppf(p, s, loc, scale):
    """
    Return quantile for given probability.
    """
    return _ppf(_cast(p), s, loc, scale)


@_overload(logpdf)
def _logpdf_ol(x, s, loc, scale):
    _type_check(logpdf, x, s, loc, scale)
    return _logpdf.__wrapped__


@_overload(pdf)
def _pdf_ol(x, s, loc, scale):
    _type_check(pdf, x, s, loc, scale)
    return _pdf.__wrapped__


@_overload(cdf)
def _cdf_ol(x, s, loc, scale):
    _type_check(ppf, x, s, loc, scale)
    return _cdf.__wrapped__


@_overload(ppf)
def _ppf_ol(p, s, loc, scale):
    _type_check(ppf, p, s, loc, scale)
    return _ppf.__wrapped__
