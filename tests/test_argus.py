import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sc

from numba_stats import argus


@pytest.mark.parametrize("chi", (0.1, 0.5, 1.0, 2.0, 3.0))
def test_logpdf(chi):
    c = 1
    p = 0.5
    x = np.linspace(-1, 2, 30)
    got = argus.logpdf(x, chi, c, p)
    expected = sc.argus.logpdf(x, chi)
    assert_allclose(got, expected)


@pytest.mark.parametrize("chi", (0.1, 0.5, 1.0, 2.0, 3.0))
def test_pdf(chi):
    c = 1
    p = 0.5
    x = np.linspace(-1, 2, 30)
    got = argus.pdf(x, chi, c, p)
    expected = sc.argus.pdf(x, chi)
    assert_allclose(got, expected)


@pytest.mark.parametrize("chi", (0.1, 0.5, 1.0, 2.0, 3.0))
def test_cdf(chi):
    c = 1
    p = 0.5
    x = np.linspace(-1, 2, 30)
    got = argus.cdf(x, chi, c, p)
    expected = sc.argus.cdf(x, chi)
    assert_allclose(got, expected, atol=2e-16)
