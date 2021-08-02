from numba_stats import qgaussian, t, norm
import numpy as np
import pytest
from scipy.integrate import quad


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


@pytest.mark.parametrize("nu", np.arange(1, 11))
def test_qgaussian_pdf_vs_t(nu):
    x = np.linspace(-5, 5)
    q, sigma = q_sigma(nu, 1.2)

    expected = t.pdf(x, nu, 0.1, 1.2)
    got = qgaussian.pdf(x, q, 0.1, sigma)

    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("q", (1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9, 2.0, 2.5, 2.9))
def test_qgaussian_cdf(q):
    x = np.linspace(-5, 5, 10)

    expected = [quad(lambda y: qgaussian.pdf(y, q, 0.1, 1.2), 0, xi)[0] for xi in x]
    got = qgaussian.cdf(x, q, 0.1, 1.2) - qgaussian.cdf(0, q, 0.1, 1.2)

    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("q", (1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9, 2.0, 2.5, 2.9))
def test_qgaussian_ppf(q):
    x = np.linspace(-5, 5, 10)

    def f(x):
        p = qgaussian.cdf(x, q, 2, 3)
        return qgaussian.ppf(p, q, 2, 3)

    expected = x
    got = f(x)

    np.testing.assert_allclose(got, expected)
