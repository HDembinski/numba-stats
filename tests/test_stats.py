import scipy.stats as sc
from scipy.integrate import quad
import numpy as np
import numba as nb


def test_njit_with_numba_stats():
    from numba_stats import norm

    @nb.njit
    def test(x):
        p = norm.cdf(x, 0, 1)
        return norm.ppf(p, 0, 1)

    expected = np.linspace(-3, 3, 10)
    got = test(expected)
    np.testing.assert_allclose(got, expected)


def test_uniform_pdf():
    from numba_stats import uniform

    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.pdf(x, -1, 2)
    expected = sc.uniform.pdf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_uniform_cdf():
    from numba_stats import uniform

    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.cdf(x, -1, 2)
    expected = sc.uniform.cdf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_uniform_ppf():
    from numba_stats import uniform

    x = np.linspace(0, 1, 10)
    got = uniform.ppf(x, -1, 2)
    expected = sc.uniform.ppf(x, -1, 2)
    np.testing.assert_allclose(got, expected)


def test_tsallis_pdf():
    from numba_stats import tsallis

    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), 0, np.inf)
                assert abs(1 - v) < err


def test_tsallis_cdf():
    from numba_stats import tsallis

    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                for ptrange in ((0, 500), (500, 1000), (1000, 2000)):
                    v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), *ptrange)
                    v2 = np.diff(tsallis.cdf(ptrange, m, t, n))
                    assert abs(v2 - v) < err
