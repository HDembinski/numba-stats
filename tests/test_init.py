def test_import():
    import numba_stats.stats as st
    import numba_stats

    for key in dir(st):
        if not key.startswith("_") and (
            key.endswith("_pdf")
            or key.endswith("_cdf")
            or key.endswith("_ppf")
            or key.endswith("_pmf")
        ):
            assert hasattr(numba_stats, key)


def test_packs():
    import numba_stats as ns
    import numba_stats.stats as st

    assert ns.norm.pdf is st.norm_pdf
    assert ns.norm.cdf is st.norm_cdf
    assert ns.norm.ppf is st.norm_ppf
