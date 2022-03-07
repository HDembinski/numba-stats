from numba_stats import uniform
import numpy as np
import scipy.stats as sc
from numpy.testing import assert_allclose


def test_pdf():
    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.pdf(x, -1, 2)
    expected = sc.uniform.pdf(x, -1, 2)
    assert_allclose(got, expected)

    got = uniform.pdf(1, -1, 3)
    expected = sc.uniform.pdf(1, -1, 3)
    assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.cdf(x, -1, 2)
    expected = sc.uniform.cdf(x, -1, 2)
    assert_allclose(got, expected)

    got = uniform.cdf(1, -1, 3)
    expected = sc.uniform.cdf(1, -1, 3)
    assert_allclose(got, expected)


def test_ppf():
    x = np.linspace(0, 1, 10)
    got = uniform.ppf(x, -1, 2)
    expected = sc.uniform.ppf(x, -1, 2)
    assert_allclose(got, expected)

    got = uniform.ppf(0.5, -1, 3)
    expected = sc.uniform.ppf(0.5, -1, 3)
    assert_allclose(got, expected)
