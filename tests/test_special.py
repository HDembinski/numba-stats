from numba_stats._special import cerf as _cerf
import numpy as np
import numba as nb
import scipy.special as sc


@nb.vectorize("complex128(complex128)")
def cerf(z):
    return _cerf(z)


def test_cerf():
    z = np.linspace(-3, 3, 10) + 1j * np.linspace(-2, 2, 10)
    expected = sc.erf(z)
    got = cerf(z)
    np.testing.assert_allclose(got, expected)
