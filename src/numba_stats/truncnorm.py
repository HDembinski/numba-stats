import numba as nb
from .norm import _pdf, _cdf, _ppf

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, xmin, xmax, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 0.0
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    return _pdf(z) * sigma_inv / (_cdf(zmax) - _cdf(zmin))


@nb.vectorize(_signatures, cache=True)
def cdf(x, xmin, xmax, mu, sigma):
    """
    Evaluate cumulative distribution function of normal distribution.
    """
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 1.0
    sigma_inv = 1 / sigma
    z = (x - mu) * sigma_inv
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    return (_cdf(z) - pmin) / (pmax - pmin)


@nb.vectorize(_signatures)
def ppf(p, xmin, xmax, mu, sigma):
    """
    Return quantile of normal distribution for given probability.
    """
    sigma_inv = 1 / sigma
    zmin = (xmin - mu) * sigma_inv
    zmax = (xmax - mu) * sigma_inv
    pmin = _cdf(zmin)
    pmax = _cdf(zmax)
    pstar = p * (pmax - pmin) + pmin
    z = _ppf(pstar)
    return sigma * z + mu
