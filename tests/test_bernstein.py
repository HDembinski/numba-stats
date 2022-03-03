from numba_stats import bernstein
from scipy.interpolate import BPoly
import pytest
import numpy as np
from scipy.integrate import quad
import numba as nb
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    "beta", [[1.0], [1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]]
)
def test_bernstein_density(beta):
    x = np.linspace(1, 3)
    got = bernstein.density(x, beta, x[0], x[-1])
    expected = BPoly(np.array(beta)[:, np.newaxis], [x[0], x[-1]])(x)
    assert_allclose(got, expected)

    got = bernstein.density(0.5, beta, 0, 1)
    expected = bernstein.density([0.5], beta, 0, 1)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1]])
def test_bernstein_integral(beta):
    xrange = 1.5, 3.4
    got = bernstein.integral(xrange[1], beta, *xrange)
    expected = np.diff(xrange)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1], [1, 2, 3], [1, 3, 2]])
def test_bernstein_integral_2(beta):
    x = np.linspace(1, 2.5)

    got = bernstein.integral(x, beta, x[0], x[-1])
    expected = [
        quad(lambda y: bernstein.density(y, beta, x[0], x[-1]), x[0], xi)[0] for xi in x
    ]
    assert_allclose(got, expected)


def test_numba_bernstein_density():
    @nb.njit
    def f():
        return bernstein.density(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            1.0,
            2.5,
        )

    assert_allclose(
        f(),
        bernstein.density(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            1.0,
            2.5,
        ),
    )
    f()


def test_numba_bernstein_integral():
    @nb.njit
    def f():
        return bernstein.integral(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            1.0,
            2.5,
        )

    assert_allclose(
        f(),
        bernstein.integral(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            1.0,
            2.5,
        ),
    )
