import numpy as np
import numba as nb
from ._special import (
    gammaln,
    erf,
    erfinv,
    xlogy,
    pdtr,
    expm1,
    log1p,
    stdtr,
    stdtrit,
    cerf,
    voigt_profile,
)


@nb.vectorize("float64(float64, float64, float64)")
def norm_pdf(x, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    z = (x - mu) / sigma
    c = 1.0 / np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) * c / sigma


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


@nb.vectorize("float64(float64, float64, float64, float64)")
def t_pdf(x, df, mu, sigma):
    """
    Return probability density of student's distribution.
    """
    z = (x - mu) / sigma
    k = 0.5 * (df + 1)
    p = np.exp(gammaln(k) - gammaln(0.5 * df))
    p /= np.sqrt(df * np.pi) * (1 + (z ** 2) / df) ** k
    return p / sigma


@nb.vectorize("float64(float64, float64, float64, float64)")
def t_cdf(x, df, mu, sigma):
    """
    Return probability density of student's distribution.
    """
    z = (x - mu) / sigma
    return stdtr(df, z)


@nb.vectorize("float64(float64, float64, float64, float64)")
def t_ppf(p, df, mu, sigma):
    """
    Return probability density of student's distribution.
    """
    if p == 0:
        return -np.inf
    elif p == 1:
        return np.inf
    z = stdtrit(df, p)
    return sigma * z + mu


@nb.vectorize("float64(float64, float64, float64, float64)")
def voigt_pdf(x, gamma, mu, sigma):
    return voigt_profile(x - mu, gamma, sigma)
