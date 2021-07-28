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
    voigt_profile,
)

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64),
]


@nb.njit
def _norm_pdf(z):
    c = 1.0 / np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) * c


@nb.njit
def _norm_cdf(z):
    c = np.sqrt(0.5)
    return 0.5 * (1.0 + erf(z * c))


@nb.njit
def _norm_ppf(p):
    return np.sqrt(2) * erfinv(2 * p - 1)


@nb.vectorize(_signatures)
def norm_pdf(x, mu, sigma):
    """
    Return probability density of normal distribution.
    """
    z = (x - mu) / sigma
    return _norm_pdf(z) / sigma


@nb.vectorize(_signatures)
def norm_cdf(x, mu, sigma):
    """
    Evaluate cumulative distribution function of normal distribution.
    """
    z = (x - mu) / sigma
    return _norm_cdf(z)


@nb.vectorize(_signatures)
def norm_ppf(p, mu, sigma):
    """
    Return quantile of normal distribution for given probability.
    """
    z = _norm_ppf(p)
    return sigma * z + mu


_signatures = [
    nb.float32(nb.int32, nb.float32),
    nb.float64(nb.intp, nb.float64),
]


@nb.vectorize(_signatures)
def poisson_pmf(k, mu):
    """
    Return probability mass for Poisson distribution.
    """
    logp = xlogy(k, mu) - gammaln(k + 1.0) - mu
    return np.exp(logp)


_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def poisson_cdf(x, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    k = np.floor(x)
    return pdtr(k, mu)


_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def cpoisson_pmf(k, mu):
    """
    Return probability mass for Poisson distribution (allow non-integer k)
    """
    logp = xlogy(k, mu) - gammaln(k + 1.0) - mu
    return np.exp(logp)


_signatures = [
    nb.float32(nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def cpoisson_cdf(x, mu):
    """
    Evaluate cumulative distribution function of Poisson distribution.
    """
    k = np.floor(x)
    return pdtr(k, mu)


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def expon_pdf(x, mu, sigma):
    """
    Return probability density of exponential distribution.
    """
    z = (x - mu) / sigma
    return np.exp(-z) / sigma


@nb.vectorize(_signatures)
def expon_cdf(x, mu, sigma):
    """
    Evaluate cumulative distribution function of exponential distribution.
    """
    z = (x - mu) / sigma
    return -expm1(-z)


@nb.vectorize(_signatures)
def expon_ppf(p, mu, sigma):
    """
    Return quantile of exponential distribution for given probability.
    """
    z = -log1p(-p)
    x = z * sigma + mu
    return x


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def t_pdf(x, df, mu, sigma):
    """
    Return probability density of student's distribution.
    """
    z = (x - mu) / sigma
    k = 0.5 * (df + 1)
    p = np.exp(gammaln(k) - gammaln(0.5 * df))
    p /= np.sqrt(df * np.pi) * (1 + (z ** 2) / df) ** k
    return p / sigma


@nb.vectorize(_signatures)
def t_cdf(x, df, mu, sigma):
    """
    Evaluate cumulative distribution function of student's distribution.
    """
    z = (x - mu) / sigma
    return stdtr(df, z)


@nb.vectorize(_signatures)
def t_ppf(p, df, mu, sigma):
    """
    Return quantile of student's distribution for given probability.
    """
    if p == 0:
        return -np.inf
    elif p == 1:
        return np.inf
    z = stdtrit(df, p)
    return sigma * z + mu


@nb.vectorize(_signatures)
def voigt_pdf(x, gamma, mu, sigma):
    """
    Return probability density of Voigtian distribution.
    """
    return voigt_profile(x - mu, gamma, sigma)


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def uniform_pdf(x, a, w):
    if a <= x <= a + w:
        return 1 / w
    return 0


@nb.vectorize(_signatures)
def uniform_cdf(x, a, w):
    if a <= x:
        if x <= a + w:
            return (x - a) / w
        return 1
    return 0


@nb.vectorize(_signatures)
def uniform_ppf(p, a, w):
    return w * p + a


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def tsallis_pdf(x, m, t, n):
    # Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    assert n > 2

    mt = np.sqrt(m ** 2 + x ** 2)
    nt = n * t
    c = (n - 1) * (n - 2) / (nt * (nt + (n - 2) * m))

    return c * x * (1 + (mt - m) / nt) ** -n


@nb.vectorize(_signatures)
def tsallis_cdf(x, m, t, n):
    # Formula computed from tsallis_pdf with Sympy, then simplified by hand
    assert n > 2

    mt = np.sqrt(m ** 2 + x ** 2)
    nt = n * t
    return ((mt - m) / nt + 1) ** (1 - n) * (m + mt - n * (mt + t)) / (m * (n - 2) + nt)


@nb.njit
def _crystalball_pdf(z, beta, m):
    assert beta > 0
    assert m > 1

    exp_beta = np.exp(-0.5 * beta ** 2)

    c = m / (beta * (m - 1.0)) * exp_beta
    # d = _norm_cdf(-beta) * np.sqrt(2 * np.pi)
    d = np.sqrt(0.5 * np.pi) * (1.0 + erf(beta * np.sqrt(0.5)))
    n = 1.0 / (c + d)

    if z <= -beta:
        a = (m / beta) ** m * exp_beta
        b = m / beta - beta
        return n * a * (b - z) ** -m
    return n * np.exp(-0.5 * z ** 2)


@nb.njit
def _crystalball_cdf(z, beta, m):
    exp_beta = np.exp(-0.5 * beta ** 2)
    c = m / (beta * (m - 1.0)) * exp_beta
    d = np.sqrt(0.5 * np.pi) * (1.0 + erf(beta * np.sqrt(0.5)))
    n = 1.0 / (c + d)

    if z <= -beta:
        return n * (
            (m / beta) ** m * exp_beta * (m / beta - beta - z) ** (1.0 - m) / (m - 1.0)
        )
    return n * (
        (m / beta) * exp_beta / (m - 1.0)
        + np.sqrt(0.5 * np.pi) * (erf(z * np.sqrt(0.5)) - erf(-beta * np.sqrt(0.5)))
    )


_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def crystalball_pdf(x, beta, m, loc, scale):
    z = (x - loc) / scale
    return _crystalball_pdf(z, beta, m) / scale


@nb.vectorize(_signatures)
def crystalball_cdf(x, beta, m, loc, scale):
    z = (x - loc) / scale
    return _crystalball_cdf(z, beta, m)


del _signatures
