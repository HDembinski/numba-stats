import numpy as np
from numba_stats import binom
import scipy.stats as sc
import pytest
import numba as nb


@pytest.mark.parametrize("n", np.linspace(0, 10, 6))
@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_pmf(n,p):
    k = np.arange(n+1)
    got = binom.pmf(k, n, p)
    expected = sc.binom.pmf(k, n, p)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("n", np.linspace(0, 10, 6))
@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_cdf(n,p):
    k = np.arange(n+1)
    got = binom.cdf(k, n, p)
    expected = sc.binom.cdf(k, n, p)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("n", np.linspace(0, 10, 6))
@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_rvs(n, p):
    got = binom.rvs(n, p, size=1000, random_state=1)

    def expected():
        np.random.seed(1)
        return np.random.binomial(n, p, 1000)

    np.testing.assert_equal(got, expected())
