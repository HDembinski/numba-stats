import importlib_metadata
import subprocess as subp

try:
    version = importlib_metadata.distribution("numba-stats").version
except importlib_metadata.PackageNotFoundError:
    # package not installed, you are developing
    version = subp.check_output(["poetry", "version", "--short"]).strip().decode()
