from numba_stats import __version__
import requests
from pkg_resources import parse_version


def test_local_against_pypi_version():
    # make sure version is up-to-date
    pypi_versions = [
        parse_version(v)
        for v in requests.get("https://pypi.org/pypi/numba-stats/json").json()[
            "releases"
        ]
    ]
    assert parse_version(__version__) not in pypi_versions, "pypi version exists"


def test_import():
    import numba_stats.stats as st
    import numba_stats

    for key in dir(st):
        if (
            key.endswith("_ppf")
            or key.endswith("_cdf")
            or key.endswith("_pdf")
            or key.endswith("_pmf")
        ):
            assert hasattr(numba_stats, key)
