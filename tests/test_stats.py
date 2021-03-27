import numba_stats.stats as nbs
import scipy.stats as sc
import scipy.special as sp
import numpy as np
import numba as nb
import warnings


def test_norm_pdf():
    x = np.linspace(-5, 5, 10)
    got = nbs.norm_pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_cdf():
    x = np.linspace(-5, 5, 10)
    got = nbs.norm_cdf(x, 1, 2)
    expected = sc.norm.cdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_ppf():
    p = np.linspace(0, 1, 10)
    got = nbs.norm_ppf(p, 0, 1)
    expected = sc.norm.ppf(p)
    np.testing.assert_allclose(got, expected)


def test_poisson_pmf():
    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = nbs.poisson_pmf(k, m)
    expected = sc.poisson.pmf(k, m)
    np.testing.assert_allclose(got, expected)


def test_poisson_cdf():
    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = nbs.poisson_cdf(k, m)
    expected = sc.poisson.cdf(k, m)
    np.testing.assert_allclose(got, expected)


def test_expon_pdf():
    x = np.linspace(1, 5, 20)
    got = nbs.expon_pdf(x, 1, 2)
    expected = sc.expon.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_expon_cdf():
    x = np.linspace(1, 5, 20) + 3
    got = nbs.expon_cdf(x, 3, 2)
    expected = sc.expon.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_expon_ppf():
    p = np.linspace(0, 1, 20)
    got = nbs.expon_ppf(p, 1, 2)
    expected = sc.expon.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_t_pdf():
    x = np.linspace(-5, 5, 10)
    got = nbs.t_pdf(x, 1.5, 2, 3)
    expected = sc.t.pdf(x, 1.5, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_t_cdf():
    x = np.linspace(-5, 5, 10)
    got = nbs.t_cdf(x, 1.5, 2, 3)
    expected = sc.t.cdf(x, 1.5, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_t_ppf():
    x = np.linspace(0, 1, 10)
    got = nbs.t_ppf(x, 1.5, 2, 3)
    expected = sc.t.ppf(x, 1.5, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_voigt_pdf():
    x = np.linspace(-5, 5, 10)
    got = nbs.voigt_pdf(x, 2, 1, 3)
    expected = sp.voigt_profile(x - 1, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_njit_with_numba_stats():
    @nb.njit
    def test(x):
        p = nbs.norm_cdf(x, 0, 1)
        return nbs.norm_ppf(p, 0, 1)

    expected = np.linspace(-3, 3, 10)
    got = test(expected)
    np.testing.assert_allclose(got, expected)
