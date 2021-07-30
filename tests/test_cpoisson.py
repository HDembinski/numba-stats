from numba_stats import poisson, cpoisson
import numpy as np


def test_cpoisson_cdf():
    mu = np.array((0.1, 0.5, 1.0, 2.0))[:, np.newaxis]
    k = np.arange(10)
    got = cpoisson.cdf(k, mu)
    expected = poisson.cdf(k, mu)
    np.testing.assert_allclose(got, expected)
