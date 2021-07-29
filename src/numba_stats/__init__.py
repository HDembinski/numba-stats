from ._version import version as __version__  # noqa

# for backward compatibility
from .stats import *  # noqa

from . import (  # noqa
    norm,
    uniform,
    poisson,
    cpoisson,
    expon,
    t,
    voigt,
    tsallis,
    crystalball,
    bernstein,
)
