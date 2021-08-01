from numba_stats import qgaussian, t, norm
import numpy as np
import pytest


def q_sigma(nu, sigma):
    # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    # relation to Student's t-distribution

    # 1 / (2 sigma^2) = 1 / (3 - q)
    # sqrt((3 - q) / 2) = sigma

    q = (nu + 3.0) / (nu + 1.0)
    return q, sigma * np.sqrt(0.5 * (3.0 - q))


def test_qgaussian_pdf_vs_norm():
    x = np.linspace(-5, 5)

    expected = norm.pdf(x, 0.5, 1.2)
    got = qgaussian.pdf(x, 1, 0.5, 1.2)

    np.testing.assert_allclose(got, expected)


def test_qgaussian_cdf_vs_norm():
    x = np.linspace(-5, 5)

    expected = norm.cdf(x, 0.5, 1.2)
    got = qgaussian.cdf(x, 1, 0.5, 1.2)

    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("nu", (1, 3, 10))
def test_qgaussian_pdf_vs_t(nu):
    x = np.linspace(-5, 5)
    q, sigma = q_sigma(nu, 1)

    expected = t.pdf(x, nu, 0, 1)
    got = qgaussian.pdf(x, q, 0, sigma)

    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("nu", (1, 3, 10))
def test_qgaussian_cdf_vs_t(nu):
    x = np.linspace(-5, 5)
    q, sigma = q_sigma(nu, 1)

    expected = t.cdf(x, nu, 0, 1)
    got = qgaussian.cdf(x, q, 0, sigma)

    np.testing.assert_allclose(got, expected)
