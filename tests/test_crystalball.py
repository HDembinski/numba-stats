from numba_stats import crystalball as cb
import numpy as np
import pytest
from scipy import stats as sc
from scipy.integrate import quad
from numpy.testing import assert_allclose


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.1, 2, 3))
def test_powerlaw_integral(beta, m):
    expected = quad(lambda z: np.exp(cb._log_powerlaw(z, beta, m)), -np.inf, -beta)[0]
    got = cb._powerlaw_integral(-beta, beta, m)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("z", (0, 1, 10, np.inf))
def test_normal_integral(beta, z):
    expected = quad(lambda z: np.exp(-0.5 * z**2), -beta, z)[0]
    got = cb._normal_integral(-beta, z)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_logpdf(beta, m):
    scale = 1.5
    x = np.linspace(-10, 5, 10)
    got = cb.logpdf(x, beta, m, 0, scale)
    expected = sc.crystalball.logpdf(x, beta, m, 0, scale)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_pdf(beta, m):
    scale = 1.5
    x = np.linspace(-10, 5, 10)
    got = cb.pdf(x, beta, m, 0, scale)
    expected = sc.crystalball.pdf(x, beta, m, 0, scale)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_cdf(beta, m):
    scale = 1.5
    x = np.linspace(-10, 5, 10)
    got = cb.cdf(x, beta, m, 0, scale)
    expected = sc.crystalball.cdf(x, beta, m, 0, scale)
    assert_allclose(got, expected)
