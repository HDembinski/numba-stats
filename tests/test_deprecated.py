import numpy as np
import numba as nb
import pytest
from numpy import VisibleDeprecationWarning


def test_deprecated_stats():
    with pytest.warns(VisibleDeprecationWarning):
        import numba_stats.stats  # noqa

    # FIXME this does not warn
    # with pytest.warns(VisibleDeprecationWarning):
    #     from numba_stats import stats  # noqa


def test_deprecated():
    import numba_stats as nbs

    from numba_stats import (
        norm,
        poisson,
        cpoisson,
        expon,
        t,
        voigt,
        uniform,
        tsallis,
        crystalball,
    )

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.norm_pdf is norm.pdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.norm_cdf is norm.cdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.norm_ppf is norm.ppf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.poisson_pmf is poisson.pmf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.cpoisson_cdf is cpoisson.cdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.expon_pdf is expon.pdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.t_pdf is t.pdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.voigt_pdf is voigt.pdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.uniform_pdf is uniform.pdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.tsallis_pdf is tsallis.pdf

    with pytest.warns(VisibleDeprecationWarning):
        assert nbs.crystalball_pdf is crystalball.pdf


def test_njit_with_numba_stats():
    import numba_stats as nbs

    @nb.njit
    def test(x):
        p = nbs.norm_cdf(x, 0, 1)
        return nbs.norm_ppf(p, 0, 1)

    expected = np.linspace(-3, 3, 10)
    with pytest.warns(VisibleDeprecationWarning):
        got = test(expected)
    np.testing.assert_allclose(got, expected)
