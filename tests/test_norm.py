import scipy.stats as sc
import numpy as np
import numba as nb
from numba_stats import norm
from numpy.testing import assert_allclose


def test_pdf_one():
    x = 1
    got = norm.pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    assert_allclose(got, expected)


def test_pdf():
    x = np.linspace(-5, 5, 10)
    got = norm.pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    assert_allclose(got, expected)


def test_logpdf():
    x = np.linspace(-5, 5, 10)
    got = norm.logpdf(x, 1, 2)
    expected = sc.norm.logpdf(x, 1, 2)
    assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 10)
    got = norm.cdf(x, 1, 2)
    expected = sc.norm.cdf(x, 1, 2)
    assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 10)
    got = norm.ppf(p, 0, 1)
    expected = sc.norm.ppf(p)
    assert_allclose(got, expected)


def test_njit():
    @nb.njit
    def test(x):
        a = norm.logpdf(x, 0, 1)
        b = norm.pdf(x, 0, 1)
        c = norm.cdf(x, 0, 1)
        d = norm.ppf(c, 0, 1)
        return a, b, c, d

    x = np.linspace(-3, 3, 10)
    a, b, c, d = test(x)

    assert_allclose(a, norm.logpdf(x, 0, 1))
    assert_allclose(b, norm.pdf(x, 0, 1))
    assert_allclose(c, norm.cdf(x, 0, 1))
    assert_allclose(d, x)
