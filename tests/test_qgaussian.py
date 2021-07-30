from numba_stats import qgaussian, t
import numpy as np
import pytest


@pytest.mark.parametrize("nu", (1, 3, 10))
def test_qgaussian_pdf(nu):
    x = np.linspace(-5, 5)

    # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    # relation to Student's t-distribution
    q = (nu + 3) / (nu + 1)
    beta = 1 / (3 - q)

    # 2 beta x^2 = (x/sigma)^2
    # 2 beta = 1/sigma^2
    # sigma = 1/sqrt(2 beta)

    expected = t.pdf(x, nu, 0, 1)
    got = qgaussian.pdf(x, q, 0, 1 / np.sqrt(2 * beta))

    np.testing.assert_allclose(got, expected)
