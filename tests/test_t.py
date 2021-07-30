import scipy.stats as sc
import numpy as np
from numba_stats import t
import pytest


@pytest.mark.parametrize("nu", (1, 3, 10))
def test_t_pdf(nu):
    x = np.linspace(-5, 5, 10)
    got = t.pdf(x, nu, 2, 3)
    expected = sc.t.pdf(x, nu, 2, 3)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("nu", (1, 3, 10))
def test_t_cdf(nu):
    x = np.linspace(-5, 5, 10)
    got = t.cdf(x, nu, 2, 3)
    expected = sc.t.cdf(x, nu, 2, 3)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("nu", (1, 3, 10))
def test_t_ppf(nu):
    x = np.linspace(0, 1, 10)
    got = t.ppf(x, nu, 2, 3)
    expected = sc.t.ppf(x, nu, 2, 3)
    np.testing.assert_allclose(got, expected)
