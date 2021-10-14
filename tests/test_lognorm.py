import scipy.stats as sc
import numpy as np


def test_lognorm_pdf():
    from numba_stats import lognorm

    x = np.linspace(0, 5, 10)
    got = lognorm.pdf(x, 1.5, 0.1, 1.2)
    expected = sc.lognorm.pdf(x, 1.5, 0.1, 1.2)
    np.testing.assert_allclose(got, expected)


def test_lognorm_logpdf():
    from numba_stats import lognorm

    x = np.linspace(0, 5, 10)
    got = lognorm.logpdf(x, 1.5, 0.1, 1.2)
    expected = sc.lognorm.logpdf(x, 1.5, 0.1, 1.2)
    np.testing.assert_allclose(got, expected)


def test_lognorm_cdf():
    from numba_stats import lognorm

    x = np.linspace(0, 5, 10)
    got = lognorm.cdf(x, 1.5, 0.1, 1.2)
    expected = sc.lognorm.cdf(x, 1.5, 0.1, 1.2)
    np.testing.assert_allclose(got, expected)


def test_lognorm_ppf():
    from numba_stats import lognorm

    p = np.linspace(0, 1, 10)
    got = lognorm.ppf(p, 1.5, 0.1, 1.2)
    expected = sc.lognorm.ppf(p, 1.5, 0.1, 1.2)
    np.testing.assert_allclose(got, expected)
