import numpy as np
from numba_stats import poisson
import scipy.stats as sc


def test_pmf():
    m = np.linspace(0, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = poisson.pmf(k, m)
    expected = sc.poisson.pmf(k, m)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    m = np.linspace(0, 3, 20)[:, np.newaxis]
    k = np.arange(10)
    got = poisson.cdf(k, m)
    expected = sc.poisson.cdf(k, m)
    np.testing.assert_allclose(got, expected)
