import scipy.stats as sc
import numpy as np


def test_norm_pdf():
    from numba_stats import norm

    x = np.linspace(-5, 5, 10)
    got = norm.pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_norm_logpdf():
    from numba_stats import norm

    x = np.linspace(-5, 5, 10)
    got = norm.logpdf(x, 1, 2)
    expected = sc.norm.logpdf(x, 1, 2)
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
