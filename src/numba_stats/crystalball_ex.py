"""
Generalised Crystal Ball distribution.

The generalised Crystal Ball distribution replaces the lower and upper tail of
an asymmetric normal distribution with power-law tails. Furthermore, the scale
is allowed to vary between the left and the right side of the peak. There is no
discontinuity at the maximum or elsewhere.
"""

from .crystalball import _powerlaw_integral, _normal_integral, _log_density
from ._util import _jit, _generate_wrappers, _prange
import numpy as np

_doc_par = """
x : Array-like
    Random variate.
beta_left : float
    Distance from the mode in units of standard deviations where the Crystal Ball
    turns from a gaussian into a power law on the left side.
m_left : float
    Absolute value of the slope of the left powerlaw tail. Must be large than 1.
scale_left : float
    Standard deviation of the left side of the mode.
beta_right : float
    Distance from the mode in units of standard deviations where the Crystal Ball
    turns from a gaussian into a power law on the right side.
m_right : float
    Absolute value of the slope of the right powerlaw tail. Must be large than 1.
scale_right : float
    Standard deviation of the right side of the mode.
loc : float
    Location of the mode of the distribution.
"""


@_jit(-3)
def _norm_half(beta, m, scale):
    T = type(beta)
    return (_powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, T(0))) * scale


@_jit(7)
def _logpdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    c = np.log(norm)
    r = np.empty_like(x)
    for i in _prange(len(r)):
        if x[i] < loc:
            beta = beta_left
            m = m_left
            z = (x[i] - loc) / scale_left
        else:
            beta = beta_right
            m = m_right
            z = (loc - x[i]) / scale_right
        r[i] = _log_density(z, beta, m) - c
    return r


@_jit(7)
def _pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    return np.exp(
        _logpdf(
            x,
            beta_left,
            m_left,
            scale_left,
            beta_right,
            m_right,
            scale_right,
            loc,
        )
    )


@_jit(7)
def _cdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    T = type(beta_left)
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    r = np.empty_like(x)
    for i in _prange(len(x)):
        scale = T(1) / (scale_left if x[i] < loc else scale_right)
        z = (x[i] - loc) * scale
        if z < -beta_left:
            r[i] = _powerlaw_integral(z, beta_left, m_left) * scale_left / norm
        elif z < 0:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, z)
                )
                * scale_left
                / norm
            )
        elif z < beta_right:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, T(0))
                )
                * scale_left
                + _normal_integral(T(0), z) * scale_right
            ) / norm
        else:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, T(0))
                )
                * scale_left
                + (
                    _normal_integral(T(0), beta_right)
                    + _powerlaw_integral(-beta_right, beta_right, m_right)
                    - _powerlaw_integral(-z, beta_right, m_right)
                )
                * scale_right
            ) / norm
    return r


_generate_wrappers(globals())
