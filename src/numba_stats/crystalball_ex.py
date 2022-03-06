"""
Generalised Crystal Ball distribution.

The generalised Crystal Ball distribution replaces the lower and upper tail of
an asymmetric normal distribution with power-law tails. Furthermore, the scale
is allowed to vary between the left and the right side of the peak. There is no
discontinuity at the maximum or elsewhere.
"""

from .crystalball import _powerlaw_integral, _normal_integral, _log_density
from ._util import _jit
import numpy as np


@_jit(-3)
def _norm_half(beta, m, scale):
    return (
        _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, type(beta)(0))
    ) * scale


@_jit(7)
def logpdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    """
    Return log of probability density.
    """
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    c = np.log(norm)
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi < loc:
            beta = beta_left
            m = m_left
            z = (xi - loc) * (type(scale_left)(1) / scale_left)
        else:
            beta = beta_right
            m = m_right
            z = (loc - xi) * (type(scale_right)(1) / scale_right)
        r[i] = _log_density(z, beta, m) - c
    return r


@_jit(7)
def pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    """
    Return probability density.
    """
    return np.exp(
        logpdf(
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
def cdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    """
    Return cumulative probability.
    """
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        scale = type(scale_left)(1) / (scale_left if xi < loc else scale_right)
        z = (xi - loc) * scale
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
                    + _normal_integral(-beta_left, type(beta_left)(0))
                )
                * scale_left
                + _normal_integral(0, z) * scale_right
            ) / norm
        else:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, type(beta_left)(0))
                )
                * scale_left
                + (
                    _normal_integral(type(beta_right)(0), beta_right)
                    + _powerlaw_integral(-beta_right, beta_right, m_right)
                    - _powerlaw_integral(-z, beta_right, m_right)
                )
                * scale_right
            ) / norm
    return r
