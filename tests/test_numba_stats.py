from numba_stats import __version__
import requests
from pkg_resources import parse_version
from pathlib import Path
import toml


def test_local_against_pypi_version():
    # make sure version is up-to-date
    pypi_versions = [
        parse_version(v)
        for v in requests.get("https://pypi.org/pypi/numba-stats/json").json()[
            "releases"
        ]
    ]
    assert parse_version(__version__) not in pypi_versions, "pypi version exists"


def test_poetry_version():
    fn = Path(__file__).parent / ".." / "pyproject.toml"
    poetry_version = toml.load(fn)["tool"]["poetry"]["version"]
    assert __version__ == poetry_version


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
