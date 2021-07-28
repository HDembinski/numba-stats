from numba_stats import ext as nbs
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
    # bernstein_density is normed so that beta corresponds to local density
    got = nbs.bernstein_density(x, beta, x[0], x[-1])
    expected = BPoly(np.array(beta)[:, np.newaxis], [x[0], x[-1]])(x) / (
        (len(beta) + 1) * (x[-1] - x[0])
    )
    np.testing.assert_allclose(got, expected)

    got = nbs.bernstein_density(0.5, beta, 0, 1)
    expected = nbs.bernstein_density([0.5], beta, 0, 1)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1], [1, 2, 3], [1, 3, 2]])
def test_bernstein_scaled_cdf(beta):
    x = np.linspace(0, 1)

    got = nbs.bernstein_scaled_cdf(x, beta, x[0], x[-1])
    expected = [
        quad(lambda y: nbs.bernstein_density(y, beta, x[0], x[-1]), x[0], xi)[0]
        for xi in x
    ]
    np.testing.assert_allclose(got, expected)


def test_numba_bernstein_density():
    @nb.njit
    def f():
        return nbs.bernstein_density(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            0.0,
            1.0,
        )

    f()


def test_numba_bernstein_scaled_cdf():
    @nb.njit
    def f():
        return nbs.bernstein_scaled_cdf(
            np.array([0.5, 0.6]),
            np.array([1.0, 2.0, 3.0]),
            0.0,
            1.0,
        )

    f()
