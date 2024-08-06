import numpy as np
from numba_stats import binom
import scipy.stats as sc
import pytest


# NC and KC are all combinations of n and k from 0 to 10
N = np.arange(10)
NC = []
KC = []
for n in N:
    for k in range(n + 1):
        NC.append(n)
        KC.append(k)
NC = np.array(NC, np.float64)
KC = np.array(KC, np.float64)


@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_pmf(p):
    print(KC, NC)
    got = binom.pmf(KC, NC, p)
    expected = sc.binom.pmf(KC, NC, p)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_cdf(p):
    got = binom.cdf(KC, NC, p)
    expected = sc.binom.cdf(KC, NC, p)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("n", np.linspace(0, 10, 6))
@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_rvs(n, p):
    got = binom.rvs(n, p, size=1000, random_state=1)

    def expected():
        np.random.seed(1)
        return np.random.binomial(n, p, 1000)

    np.testing.assert_equal(got, expected())
