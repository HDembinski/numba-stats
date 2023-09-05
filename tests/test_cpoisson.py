from numba_stats import poisson, cpoisson
import numpy as np
import numba as nb
import pytest


@pytest.mark.parametrize("mu", (0.1, 0.5, 1.0, 2.0))
def test_cdf(mu):
    k = np.arange(10)
    got = cpoisson.cdf(k, mu)
    expected = poisson.cdf(k, mu)
    np.testing.assert_allclose(got, expected)


def test_njit():
    mu = 0.5

    @nb.njit
    def f(k):
        return cpoisson.cdf(k, mu)

    k = np.arange(10)
    np.testing.assert_allclose(f(k), cpoisson.cdf(k, mu))

    k = 10
    np.testing.assert_allclose(f(k), cpoisson.cdf(k, mu))
