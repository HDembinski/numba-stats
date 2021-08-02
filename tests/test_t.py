import scipy.stats as sc
import numpy as np
from numba_stats import t
import pytest


@pytest.mark.parametrize("df", (1, 1.5, 2, 3, 4, 5.5, 10))
def test_t_pdf(df):
    x = np.linspace(-5, 5, 10)
    got = t.pdf(x, df, 2, 3)
    expected = sc.t.pdf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_t_cdf(df):
    x = np.linspace(-5, 5, 10)
    got = t.cdf(x, df, 2, 3)
    expected = sc.t.cdf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_t_ppf(df):
    x = np.linspace(0, 1, 10)
    got = t.ppf(x, df, 2, 3)
    expected = sc.t.ppf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)
