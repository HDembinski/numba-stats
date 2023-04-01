from numba_stats import cruijff
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose
import math


def test_pdf():
    x = np.linspace(-5, 5, 100)
    got = cruijff.pdf(x, 0, 1, 1, 0, 0)
    expected = norm().pdf(x) * math.sqrt(2 * math.pi)  # scale by root(2pi)
    assert_allclose(got, expected)
