"""
Noxfile to orchestrate tests and computing coverage.

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

nox.options.sessions = ["test", "maxtest"]

# running in parallel with pytest-xdist does not make the tests faster


@nox.session(reuse_venv=True)
def test(session: nox.Session) -> None:
    """Run all tests."""
    session.install("-e.[test]")
    session.run("pytest", *session.posargs)


@nox.session(python=LATEST_PYTHON, reuse_venv=True)
def maxtest(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.[test]")
    session.run("pytest", *session.posargs, env=ENV)


# Python-3.12 provides coverage info faster
@nox.session(python="3.12", reuse_venv=True)
def cov(session: nox.Session) -> None:
    """Run covage and place in 'htmlcov' directory."""
    import warnings

    warnings.warn("JIT-compiled code lines are not counted.")

    session.install("-e.[test]")
    session.run("coverage", "run", "-m", "pytest", env=ENV)
    session.run("coverage", "html", "-d", "htmlcov")
    session.run("coverage", "report", "-m")
