from numba_stats import cruijff
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose


def test_density_1():
    x = np.linspace(-5, 5, 100)
    got = cruijff.density(x, 0, 0, 0, 1, 1) / np.sqrt(2 * np.pi)
    expected = norm().pdf(x)
    assert_allclose(got, expected)


def test_density_2():
    x = np.linspace(-5, 5, 100)
    got = cruijff.density(x, 0.1, 0.2, 1.5, 2.1, 2.1)
    expected = cruijff.density((x - 1.5) / 2.1, 0.1, 0.2, 0, 1, 1)
    assert_allclose(got, expected)


def test_density_3():
    x = np.linspace(-5, 5, 100)
    y = cruijff.density(x, 0.2, 0.3, 0, 1.1, 1.2)
    dy = np.diff(y, 1)
    # density should be smooth
    assert_allclose(dy, 0, atol=0.05)
    # function rises up to x = 0
    assert np.all(dy[x[1:] < 0] > 0)
    # function falls after x = 0
    assert np.all(dy[x[1:] > 0.1] < 0)
