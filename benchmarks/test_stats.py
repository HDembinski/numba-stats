from numba_stats.stats import norm_pdf, norm_cdf, norm_ppf, poisson_pmf, poisson_cdf
import scipy.stats as sc
import numpy as np
import pytest


@pytest.mark.benchmark(group="norm_pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_pdf_speed(benchmark, which, n):
    x = np.linspace(-5, 5, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.norm.pdf(x, m, s) if which == "scipy" else norm_pdf(x, m, s))


@pytest.mark.benchmark(group="norm_cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_cdf_speed(benchmark, which, n):
    x = np.linspace(-5, 5, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.norm.cdf(x, m, s) if which == "scipy" else norm_cdf(x, m, s))


@pytest.mark.benchmark(group="norm_ppf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_ppf_speed(benchmark, which, n):
    x = np.linspace(0, 1, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.norm.ppf(x, m, s) if which == "scipy" else norm_ppf(x, m, s))


@pytest.mark.benchmark(group="poisson_pmf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_poisson_pmf_speed(benchmark, which, n):
    k = np.arange(0, n)
    m = np.linspace(0, 10, n)
    benchmark(lambda: sc.poisson.pmf(k, m) if which == "scipy" else poisson_pmf(k, m))


@pytest.mark.benchmark(group="poisson_cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_poisson_cdf_speed(benchmark, which, n):
    k = np.arange(0, n)
    m = np.linspace(0, 10, n)
    benchmark(lambda: sc.poisson.cdf(k, m) if which == "scipy" else poisson_cdf(k, m))
