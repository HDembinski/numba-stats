from numba_stats import _special as sp
from scipy import special as sp_ref
import numba as nb
import pytest
import numpy as np


@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("b", [1, 2, 3])
@pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
def test_betainc(a, b, x):
    @nb.njit
    def betainc(a, b, x):
        return sp.betainc(a, b, x)

    np.testing.assert_allclose(betainc(a, b, x), sp_ref.betainc(a, b, x))
