"""
Generalised Crystal Ball distribution.

The generalised Crystal Ball distribution replaces the lower and upper tail of
an asymmetric normal distribution with power-law tails. Furthermore, the scale
is allowed to vary between the left and the right side of the peak. There is no
discontinuity at the maximum or elsewhere.
"""

from .crystalball import _powerlaw_integral, _normal_integral, _log_density
from ._util import _vectorize, _jit
import numpy as np


@_jit
def _norm_half(beta, m, scale):
    return (_normal_integral(0, beta) + _powerlaw_integral(-beta, beta, m)) * scale


@_jit
def _logpdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    if x < loc:
        inv_scale = 1 / scale_left
        beta = beta_left
        m = m_left
        z = x - loc
    else:
        inv_scale = 1 / scale_right
        beta = beta_right
        m = m_right
        z = loc - x
    z *= inv_scale
    log_dens = _log_density(z, beta, m)
    return log_dens - np.log(
        _norm_half(beta_left, m_left, scale_left)
        + _norm_half(beta_right, m_right, scale_right)
    )


@_vectorize(8)
def logpdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    """
    Return log of probability density.
    """
    return _logpdf(
        x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc
    )


@_vectorize(8)
def pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    """
    Return probability density.
    """
    return np.exp(
        _logpdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
    )


@_vectorize(8)
def cdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    """
    Return cumulative probability.
    """
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    z = x - loc
    if z < 0:
        z /= scale_left
    else:
        z /= scale_right
    if z < -beta_left:
        return _powerlaw_integral(z, beta_left, m_left) / norm * scale_left
    if z < 0:
        return (
            (
                _powerlaw_integral(-beta_left, beta_left, m_left)
                + _normal_integral(-beta_left, z)
            )
            * scale_left
            / norm
        )
    if z < beta_right:
        return (
            (
                _powerlaw_integral(-beta_left, beta_left, m_left)
                + _normal_integral(-beta_left, 0)
            )
            * scale_left
            + _normal_integral(0, z) * scale_right
        ) / norm
    return (
        (
            _powerlaw_integral(-beta_left, beta_left, m_left)
            + _normal_integral(-beta_left, 0)
        )
        * scale_left
        + (
            _normal_integral(0, beta_right)
            + _powerlaw_integral(-beta_right, beta_right, m_right)
            - _powerlaw_integral(-z, beta_right, m_right)
        )
        * scale_right
    ) / norm
