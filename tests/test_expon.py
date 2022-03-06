import scipy.stats as sc
import numpy as np
from numba_stats import expon


def test_pdf():
    x = np.linspace(1, 5, 20)
    got = expon.pdf(x, 1, 2)
    expected = sc.expon.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(1, 5, 20) + 3
    got = expon.cdf(x, 3, 2)
    expected = sc.expon.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 20)
    got = expon.ppf(p, 1, 2)
    expected = sc.expon.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)
