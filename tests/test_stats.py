import numba_stats.stats as nb
import scipy.stats as sc
import numpy as np


def test_norm_pdf():
    x = np.linspace(-5, 5, 10)
    got = nb.norm_pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_cdf():
    x = np.linspace(-5, 5, 10)
    got = nb.norm_cdf(x, 1, 2)
    expected = sc.norm.cdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_ppf():
    p = np.linspace(0, 1, 10)
    got = nb.norm_ppf(p, 0, 1)
    expected = sc.norm.ppf(p)
    np.testing.assert_allclose(got, expected)


def test_poisson_pmf():
    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = nb.poisson_pmf(k, m)
    expected = sc.poisson.pmf(k, m)
    np.testing.assert_allclose(got, expected)


def test_poisson_cdf():
    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = nb.poisson_cdf(k, m)
    expected = sc.poisson.cdf(k, m)
    np.testing.assert_allclose(got, expected)


def test_expon_pdf():
    x = np.linspace(1, 5, 20)
    got = nb.expon_pdf(x, 1, 2)
    expected = sc.expon.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_expon_cdf():
    x = np.linspace(1, 5, 20) + 3
    got = nb.expon_cdf(x, 3, 2)
    expected = sc.expon.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_expon_ppf():
    p = np.linspace(0, 1, 20)
    got = nb.expon_ppf(p, 1, 2)
    expected = sc.expon.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)
