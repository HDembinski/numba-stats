import warnings
from numpy import VisibleDeprecationWarning

warnings.warn(
    """numba_stats.stats submodule will be removed in v1.0
Please import distributions like this:

from numba_stats import norm

norm.pdf(1, 2, 3)
""",
    VisibleDeprecationWarning,
    1,
)

from .norm import pdf as norm_pdf, cdf as norm_cdf, ppf as norm_ppf  # noqa
from .poisson import pmf as poisson_pmf, cdf as poisson_cdf  # noqa
from .cpoisson import cdf as cpoisson_cdf  # noqa
from .expon import pdf as expon_pdf, cdf as expon_cdf, ppf as expon_ppf  # noqa
from .crystalball import pdf as crystalball_pdf, cdf as crystalball_cdf  # noqa
from .t import pdf as t_pdf, cdf as t_cdf, ppf as t_ppf  # noqa
from .tsallis import pdf as tsallis_pdf, cdf as tsallis_cdf  # noqa
from .uniform import pdf as uniform_pdf, cdf as uniform_cdf, ppf as uniform_ppf  # noqa
from .voigt import pdf as voigt_pdf  # noqa
