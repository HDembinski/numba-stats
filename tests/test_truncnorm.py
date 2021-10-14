import numpy as np
from numba_stats import truncnorm, norm


def test_pdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    got = truncnorm.pdf(x, xmin, xmax, mu, sigma)
    expected = norm.pdf(x, mu, sigma) / (
        norm.cdf(xmax, mu, sigma) - norm.cdf(xmin, mu, sigma)
    )
    expected[x < xmin] = 0
    expected[x > xmax] = 0
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    got = truncnorm.cdf(x, xmin, xmax, mu, sigma)
    expected = (norm.cdf(x, mu, sigma) - norm.cdf(xmin, mu, sigma)) / (
        norm.cdf(xmax, mu, sigma) - norm.cdf(xmin, mu, sigma)
    )
    expected[x < xmin] = 0
    expected[x > xmax] = 1
    np.testing.assert_allclose(got, expected)


def test_ppf():
    expected = np.linspace(0, 1, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    x = truncnorm.ppf(expected, mu, sigma, xmin, xmax)
    got = truncnorm.cdf(x, mu, sigma, xmin, xmax)
    np.testing.assert_allclose(got, expected, atol=1e-14)
