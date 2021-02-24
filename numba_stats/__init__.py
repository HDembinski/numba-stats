from importlib_metadata import distribution as _d

__version__ = _d("numba-stats").version

from .stats import (  # noqa
    norm_pdf,
    norm_cdf,
    norm_ppf,
    poisson_pmf,
    poisson_cdf,
    expon_pdf,
    expon_cdf,
    expon_ppf,
    t_pdf,
    t_cdf,
    t_ppf,
    voigt_pdf,
)
