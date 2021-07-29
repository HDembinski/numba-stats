from ._version import version as __version__  # noqa

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
)

# from .not_in_scipy import bernstein_density, bernstein_scaled_cdf  # noqa
# bernstein = Namespace(density=bernstein_density, scaled_cdf=bernstein_scaled_cdf)
