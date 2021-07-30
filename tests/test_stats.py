import scipy.stats as sc
import scipy.special as sp
from scipy.integrate import quad
import numpy as np
import numba as nb
import pytest


def test_norm_pdf():
    from numba_stats import norm

    x = np.linspace(-5, 5, 10)
    got = norm.pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_cdf():
    from numba_stats import norm

    x = np.linspace(-5, 5, 10)
    got = norm.cdf(x, 1, 2)
    expected = sc.norm.cdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_ppf():
    from numba_stats import norm

    p = np.linspace(0, 1, 10)
    got = norm.ppf(p, 0, 1)
    expected = sc.norm.ppf(p)
    np.testing.assert_allclose(got, expected)


def test_poisson_pmf():
    from numba_stats import poisson

    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = poisson.pmf(k, m)
    expected = sc.poisson.pmf(k, m)
    np.testing.assert_allclose(got, expected)


def test_poisson_cdf():
    from numba_stats import poisson

    m = np.linspace(0.1, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = poisson.cdf(k, m)
    expected = sc.poisson.cdf(k, m)
    np.testing.assert_allclose(got, expected)


def test_expon_pdf():
    from numba_stats import expon

    x = np.linspace(1, 5, 20)
    got = expon.pdf(x, 1, 2)
    expected = sc.expon.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_expon_cdf():
    from numba_stats import expon

    x = np.linspace(1, 5, 20) + 3
    got = expon.cdf(x, 3, 2)
    expected = sc.expon.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_expon_ppf():
    from numba_stats import expon

    p = np.linspace(0, 1, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        got = expon.ppf(p, 1, 2)
    expected = sc.expon.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_voigt_pdf():
    from numba_stats import voigt

    x = np.linspace(-5, 5, 10)
    got = voigt.pdf(x, 2, 1, 3)
    expected = sp.voigt_profile(x - 1, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_njit_with_numba_stats():
    from numba_stats import norm

    @nb.njit
    def test(x):
        p = norm.cdf(x, 0, 1)
        return norm.ppf(p, 0, 1)

    expected = np.linspace(-3, 3, 10)
    got = test(expected)
    np.testing.assert_allclose(got, expected)


def test_uniform_pdf():
    from numba_stats import uniform

    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.pdf(x, -1, 2)
    expected = sc.uniform.pdf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_uniform_cdf():
    from numba_stats import uniform

    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.cdf(x, -1, 2)
    expected = sc.uniform.cdf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_uniform_ppf():
    from numba_stats import uniform

    x = np.linspace(0, 1, 10)
    got = uniform.ppf(x, -1, 2)
    expected = sc.uniform.ppf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_tsallis_pdf():
    from numba_stats import tsallis

    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), 0, np.inf)
                assert abs(1 - v) < err


def test_tsallis_cdf():
    from numba_stats import tsallis

    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                for ptrange in ((0, 500), (500, 1000), (1000, 2000)):
                    v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), *ptrange)
                    v2 = np.diff(tsallis.cdf(ptrange, m, t, n))
                    assert abs(v2 - v) < err


@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_crystalball_pdf(m):
    from numba_stats import crystalball

    x = np.linspace(-10, 5, 10)
    beta = 1
    got = crystalball.pdf(x, beta, m, 0, 1)
    expected = sc.crystalball.pdf(x, beta, m)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_crystalball_cdf(m):
    from numba_stats import crystalball

    x = np.linspace(-10, 5, 10)
    beta = 1
    got = crystalball.cdf(x, beta, m, 0, 1)
    expected = sc.crystalball.cdf(x, beta, m)
    np.testing.assert_allclose(got, expected)
