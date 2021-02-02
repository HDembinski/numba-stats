import numpy as np
import numba as nb
from ._special import gammaln, erf, erfinv, xlogy, pdtr, expm1, log1p


@nb.vectorize("float64(float64, float64, float64)")
def norm_pdf(x, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    z = (x - mu) / sigma
    c = 1.0 / np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) / sigma * c


@nb.vectorize("float64(float64, float64, float64)")
def norm_cdf(x, mu, sigma):
    """
    Evaluate cumulative distribution function of normal distribution.
    """
    z = (x - mu) / sigma
    z *= 1.0 / np.sqrt(2)
    return 0.5 * (1.0 + erf(z))


@nb.vectorize("float64(float64, float64, float64)")
def norm_ppf(p, mu, sigma):
    """
    Return quantile of normal distribution for given probability.
    """
    z = np.sqrt(2) * erfinv(2 * p - 1)
    return sigma * z + mu


@nb.vectorize("float64(intp, float64)")
def poisson_pmf(k, mu):
    """
    Return probability mass for Poisson distribution.
    """
    logp = xlogy(k, mu) - gammaln(k + 1.0) - mu
    return np.exp(logp)


@nb.vectorize("float64(intp, float64)")
def poisson_cdf(x, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    k = np.floor(x)
    return pdtr(k, mu)


@nb.vectorize("float64(float64, float64, float64)")
def expon_pdf(x, mu, sigma):
    """
    Return probability density of exponential distribution.
    """
    z = (x - mu) / sigma
    return np.exp(-z) / sigma


@nb.vectorize("float64(float64, float64, float64)")
def expon_cdf(x, mu, sigma):
    """
    Evaluate cumulative distribution function of exponential distribution.
    """
    z = (x - mu) / sigma
    return -expm1(-z)


@nb.vectorize("float64(float64, float64, float64)")
def expon_ppf(p, mu, sigma):
    """
    Return quantile of exponential distribution for given probability.
    """
    z = -log1p(-p)
    x = z * sigma + mu
    return x
