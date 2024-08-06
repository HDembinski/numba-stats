"""
Noxfile for iminuit.

Pass extra arguments to pytest after --
"""

import nox
import sys

sys.path.append(".")
import python_releases

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"

ENV = {
    "COVERAGE_CORE": "sysmon",  # faster coverage on Python 3.12
}

PYPROJECT = nox.project.load_toml("pyproject.toml")
MINIMUM_PYTHON = PYPROJECT["project"]["requires-python"].strip(">=")
LATEST_PYTHON = str(python_releases.latest())

nox.options.sessions = ["test", "mintest", "maxtest"]


@nox.session(reuse_venv=True)
def test(session: nox.Session) -> None:
    """Run all tests."""
    # running in parallel with pytest-xdist crashes ROOT
    session.install("-e.[test]")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args)


# broken: this tries to compile dependencies
@nox.session(python=MINIMUM_PYTHON, venv_backend="uv", reuse_venv=True)
def mintest(session: nox.Session) -> None:
    """Run tests on the minimum python version."""
    session.install("-ve.[test]", "--only-binary=1", "--resolution=lowest-direct")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args)


@nox.session(python=LATEST_PYTHON, reuse_venv=True)
def maxtest(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-ve.[test]", "--only-binary=1")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args, env=ENV)


# Python-3.12 provides coverage info faster
@nox.session(python="3.12", reuse_venv=True)
def cov(session: nox.Session) -> None:
    """Run covage and place in 'htmlcov' directory."""
    session.install("-e.[test]")
    session.run("coverage", "run", "-m", "pytest", env=ENV)
    session.run("coverage", "html", "-d", "build/htmlcov")
    session.run("coverage", "report", "-m")
