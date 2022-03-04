"""
Normal distribution.
"""
import numpy as np
from numba.extending import overload as _overload
from ._special import erfinv as _erfinv
from ._util import _jit, _cast, _trans, _type_check
from math import erf as _erf


@_jit(-1)
def _logpdfz(z):
    return -0.5 * (z**2 + np.log(2 * np.pi))


@_jit(-1)
def _cdfz(z):
    c = np.sqrt(0.5)
    return 0.5 * (1.0 + _erf(z * c))


@_jit(-1, cache=False)  # cannot cache because of _erfinv
def _ppfz(p):
    return np.sqrt(2) * _erfinv(2 * p - 1)


@_jit(2)
def _logpdf(x, loc, scale):
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        r[i] = _logpdfz(ri) - np.log(scale)
    return r


@_jit(2)
def _pdf(x, loc, scale):
    return np.exp(_logpdf(x, loc, scale))


@_jit(2)
def _cdf(x, loc, scale):
    r = _trans(x, loc, scale)
    for i, ri in enumerate(r):
        r[i] = _cdfz(ri)
    return r


@_jit(2, cache=False)
def _ppf(p, loc, scale):
    r = np.empty_like(p)
    for i in range(len(p)):
        r[i] = scale * _ppfz(p[i]) + loc
    return r


def logpdf(x, loc, scale):
    """
    Return log of probability density.
    """
    return _logpdf(_cast(x), loc, scale)


def pdf(x, loc, scale):
    """
    Return probability density.
    """
    return _pdf(_cast(x), loc, scale)


def cdf(x, loc, scale):
    """
    Return cumulative probability.
    """
    return _cdf(x, loc, scale)


def ppf(p, loc, scale):
    """
    Return quantile for given probability.
    """
    return _ppf(_cast(p), loc, scale)


@_overload(logpdf)
def _logpdf_ol(x, loc, scale):
    _type_check(logpdf, x, loc, scale)
    return _logpdf.__wrapped__


@_overload(pdf)
def _pdf_ol(x, loc, scale):
    _type_check(pdf, x, loc, scale)
    return _pdf.__wrapped__


@_overload(cdf)
def _cdf_ol(x, loc, scale):
    _type_check(ppf, x, loc, scale)
    return _cdf.__wrapped__


@_overload(ppf)
def _ppf_ol(p, loc, scale):
    _type_check(ppf, p, loc, scale)
    return _ppf.__wrapped__
