from numba_stats import bernstein
from scipy.interpolate import BPoly
import pytest
import numpy as np
from scipy.integrate import quad
import numba as nb


@pytest.mark.parametrize(
    "beta", [[1.0], [1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]]
)
def test_bernstein_density(beta):
    x = np.linspace(1, 3)
    got = bernstein.density(x, beta, x[0], x[-1])
    expected = BPoly(np.array(beta)[:, np.newaxis], [x[0], x[-1]])(x)
    np.testing.assert_allclose(got, expected)

    got = bernstein.density(0.5, beta, 0, 1)
    expected = bernstein.density([0.5], beta, 0, 1)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1]])
def test_bernstein_integral(beta):
    xrange = 1.5, 3.4
    got = bernstein.scaled_cdf(xrange[1], beta, *xrange)
    expected = 1
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1], [1, 2, 3], [1, 3, 2]])
def test_bernstein_scaled_cdf(beta):
    x = np.linspace(0, 1)

    got = bernstein.scaled_cdf(x, beta, x[0], x[-1])
    expected = [
        quad(lambda y: bernstein.density(y, beta, x[0], x[-1]), x[0], xi)[0] for xi in x
    ]
    np.testing.assert_allclose(got, expected)


def test_numba_bernstein_density():
    @nb.njit
    def f():
        return bernstein.density(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            0.0,
            1.0,
        )

    f()


def test_numba_bernstein_scaled_cdf():
    @nb.njit
    def f():
        return bernstein.scaled_cdf(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            0.0,
            1.0,
        )

    f()
