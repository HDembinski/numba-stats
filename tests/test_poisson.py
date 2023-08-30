import numpy as np
from numba_stats import poisson
import scipy.stats as sc
import pytest


@pytest.mark.parametrize("mu", np.linspace(0, 3, 5))
def test_pmf(mu):
    k = np.arange(10)
    got = poisson.pmf(k, mu)
    expected = sc.poisson.pmf(k, mu)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("mu", np.linspace(0, 3, 5))
def test_cdf(mu):
    k = np.arange(10)
    got = poisson.cdf(k, mu)
    expected = sc.poisson.cdf(k, mu)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("func", ("cdf", "pmf", "logpmf"))
def test_broadcasting(func):
    k = np.arange(10)
    mu = np.array([4, 10, 15])

    got = getattr(poisson, func)(k, mu[:, np.newaxis])
    expected = getattr(sc.poisson, func)(k, mu[:, np.newaxis])
    np.testing.assert_allclose(got, expected)
