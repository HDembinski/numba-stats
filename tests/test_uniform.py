from numba_stats import uniform
import numpy as np
import scipy.stats as sc


def test_pdf():
    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.pdf(x, -1, 2)
    expected = sc.uniform.pdf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.cdf(x, -1, 2)
    expected = sc.uniform.cdf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    x = np.linspace(0, 1, 10)
    got = uniform.ppf(x, -1, 2)
    expected = sc.uniform.ppf(x, -1, 2)
    np.testing.assert_allclose(got, expected)
