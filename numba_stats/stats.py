import numpy as np
import numba as nb
from ._special import gammaincc, gamma, erf, erfinv


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
    return mu ** k * np.exp(-mu) / gamma(k + 1)


@nb.vectorize("float64(intp, float64)")
def poisson_cdf(k, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    return gammaincc(k + 1, mu)


@nb.vectorize("float64(float64, float64)")
def expon_pdf(x, lambd):
    """
    Return probability mass for Poisson distribution.
    """
    return lambd * np.exp(-lambd * x)


@nb.vectorize("float64(intp, float64)")
def expon_cdf(k, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    return gammaincc(k + 1, mu)
