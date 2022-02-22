from .crystalball import _powerlaw_integral, _normal_integral, _density
import numba as nb


@nb.njit
def _norm(beta_left, m_left, beta_right, m_right, scale_left, scale_right):
    return (
        _powerlaw_integral(-beta_left, beta_left, m_left)
        + _normal_integral(-beta_left, 0)
    ) * scale_left + (
        _normal_integral(0, beta_right)
        + _powerlaw_integral(-beta_right, beta_right, m_right)
    ) * scale_right


@nb.vectorize(cache=True)
def pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    if x < loc:
        scale = scale_left
        beta = beta_left
        m = m_left
        z = (x - loc) / scale
    else:
        scale = scale_right
        beta = beta_right
        m = m_right
        z = (loc - x) / scale
    dens = _density(z, beta, m)
    return dens / _norm(beta_left, m_left, beta_right, m_right, scale_left, scale_right)


@nb.vectorize(cache=True)
def cdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    norm = _norm(beta_left, m_left, beta_right, m_right, scale_left, scale_right)
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
