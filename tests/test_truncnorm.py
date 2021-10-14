import scipy.stats as sc
import numpy as np
from numba_stats import truncnorm


def test_pdf():
    x = np.linspace(-1, 5, 10)
    xmin = 0
    xmax = 4
    mu = 1
    sigma = 2
    got = truncnorm.pdf(x, mu, sigma, xmin, xmax)
    z = (x - mu) / sigma
    zmin = (xmin - mu) / sigma
    zmax = (xmax - mu) / sigma
    expected = sc.truncnorm.pdf(z, zmin, zmax) / sigma
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1, 5, 10)
    xmin = 0
    xmax = 4
    mu = 1
    sigma = 2
    got = truncnorm.cdf(x, mu, sigma, xmin, xmax)
    z = (x - mu) / sigma
    zmin = (xmin - mu) / sigma
    zmax = (xmax - mu) / sigma
    expected = sc.truncnorm.cdf(z, zmin, zmax)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 10)
    xmin = 0
    xmax = 4
    mu = 1
    sigma = 2
    got = truncnorm.ppf(p, mu, sigma, xmin, xmax)
    zmin = (xmin - mu) / sigma
    zmax = (xmax - mu) / sigma
    expected = sc.truncnorm.ppf(p, zmin, zmax) * sigma + mu
    np.testing.assert_allclose(got, expected, atol=1e-14)
