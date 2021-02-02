from numba_stats.stats import norm_pdf, norm_cdf, norm_ppf, poisson_pmf, poisson_cdf
import scipy.stats as sc
import numpy as np


def test_norm_pdf():
    x = np.linspace(-5, 5, 10)
    got = norm_pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_cdf():
    x = np.linspace(-5, 5, 10)
    got = norm_cdf(x, 1, 2)
    expected = sc.norm.cdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_ppf():
    p = np.linspace(0, 1, 10)
    got = norm_ppf(p, 0, 1)
    expected = sc.norm.ppf(p)
    np.testing.assert_allclose(got, expected)


def test_poisson_pmf():
    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = poisson_pmf(k, m)
    expected = sc.poisson.pmf(k, m)
    np.testing.assert_allclose(got, expected)


def test_poisson_cdf():
    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = poisson_cdf(k, m)
    expected = sc.poisson.cdf(k, m)
    np.testing.assert_allclose(got, expected)
