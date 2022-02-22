from numba_stats import crystalball_ex as cb
import numpy as np
import pytest
from scipy import stats as sc
from numpy.testing import assert_allclose
from scipy.integrate import quad


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_pdf_left(beta, m):
    scale = 1.5
    x = np.linspace(-10, 0, 10)
    got = cb.pdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, 0)

    max1 = cb.pdf(0, beta, m, scale, 2 * beta, 2 * m, 2 * scale, 0)
    max2 = sc.crystalball.pdf(0, beta, m, 0, scale)

    expected = sc.crystalball.pdf(x, beta, m, 0, scale) * max1 / max2
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_pdf_right(beta, m):
    scale = 1.5
    x = np.linspace(-10, 0, 10)
    got = cb.pdf(-x, 2 * beta, 2 * m, 2 * scale, beta, m, scale, 0)

    max1 = cb.pdf(0, beta, m, scale, 2 * beta, 2 * m, 2 * scale, 0)
    max2 = sc.crystalball.pdf(0, beta, m, 0, scale)

    expected = sc.crystalball.pdf(x, beta, m, 0, scale) * max1 / max2
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_pdf_integral(beta, m):
    scale = 1.5
    got = quad(
        lambda x: cb.pdf(x, 2 * beta, 2 * m, 2 * scale, beta, m, scale, 0),
        -np.inf,
        np.inf,
    )[0]
    assert_allclose(got, 1)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
def test_cdf(beta, m):
    scale = 1.5
    x = np.linspace(-10, 10, 10)
    got = cb.cdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, 0)
    expected = [
        quad(
            lambda x: cb.pdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, 0),
            -np.inf,
            xi,
        )[0]
        for xi in x
    ]
    assert_allclose(got, expected)
