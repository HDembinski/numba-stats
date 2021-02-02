from numba_stats import __version__
import requests
from pkg_resources import parse_version
from pathlib import Path
import toml


def test_local_against_pypi_version():
    # make sure version is up-to-date
    pypi_versions = [
        parse_version(v)
        for v in requests.get("https://pypi.org/pypi/iminuit/json").json()["releases"]
    ]
    assert parse_version(__version__) not in pypi_versions, "pypi version exists"


def test_poetry_version():
    fn = Path(__file__).parent / ".." / "pyproject.toml"
    poetry_version = toml.load(fn)["tool"]["poetry"]["version"]
    assert __version__ == poetry_version
