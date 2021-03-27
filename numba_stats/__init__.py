from ._version import version as __version__  # noqa

from .stats import (  # noqa
    uniform_pdf,
    uniform_cdf,
    uniform_ppf,
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

from argparse import Namespace

uniform = Namespace(pdf=uniform_pdf, cdf=uniform_cdf, ppf=uniform_ppf)
norm = Namespace(pdf=norm_pdf, cdf=norm_cdf, ppf=norm_ppf)
poisson = Namespace(pmf=poisson_pmf, cdf=poisson_cdf)
expon = Namespace(pdf=expon_pdf, cdf=expon_cdf, ppf=expon_ppf)
t = Namespace(pdf=t_pdf, cdf=t_cdf, ppf=t_ppf)
voigt = Namespace(pdf=voigt_pdf)

del Namespace
