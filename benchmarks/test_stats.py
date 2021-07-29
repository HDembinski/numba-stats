import numba_stats as nb
import scipy.stats as sc
import numpy as np
import pytest


@pytest.mark.benchmark(group="norm.pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_pdf_speed(benchmark, which, n):
    x = np.linspace(-5, 5, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(
        lambda: sc.norm.pdf(x, m, s) if which == "scipy" else nb.norm.pdf(x, m, s)
    )


@pytest.mark.benchmark(group="norm.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_cdf_speed(benchmark, which, n):
    x = np.linspace(-5, 5, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(
        lambda: sc.norm.cdf(x, m, s) if which == "scipy" else nb.norm.cdf(x, m, s)
    )


@pytest.mark.benchmark(group="norm.ppf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_ppf_speed(benchmark, which, n):
    x = np.linspace(0, 1, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(
        lambda: sc.norm.ppf(x, m, s) if which == "scipy" else nb.norm.ppf(x, m, s)
    )


@pytest.mark.benchmark(group="poisson.pmf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_poisson_pmf_speed(benchmark, which, n):
    k = np.arange(0, n)
    m = np.linspace(0, 10, n)
    benchmark(
        lambda: sc.poisson.pmf(k, m) if which == "scipy" else nb.poisson.pmf(k, m)
    )


@pytest.mark.benchmark(group="poisson.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_poisson_cdf_speed(benchmark, which, n):
    k = np.arange(0, n)
    m = np.linspace(0, 10, n)
    benchmark(
        lambda: sc.poisson.cdf(k, m) if which == "scipy" else nb.poisson.cdf(k, m)
    )


@pytest.mark.benchmark(group="expon.pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_expon_pdf_speed(benchmark, which, n):
    x = np.linspace(0, 10, n)
    m = np.linspace(0, 10, n)
    s = np.linspace(1, 10, n)
    benchmark(
        lambda: sc.expon.pdf(x, m, s) if which == "scipy" else nb.expon.pdf(x, m, s)
    )


@pytest.mark.benchmark(group="expon.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_expon_cdf_speed(benchmark, which, n):
    x = np.linspace(0, 10, n)
    m = np.linspace(0, 10, n)
    s = np.linspace(1, 10, n)
    benchmark(
        lambda: sc.expon.cdf(x, m, s) if which == "scipy" else nb.expon.cdf(x, m, s)
    )


@pytest.mark.benchmark(group="t.pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_t_pdf_speed(benchmark, which, n):
    x = np.linspace(-5, 5, n)
    df = np.linspace(1, 10, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(
        lambda: sc.t.pdf(x, df, m, s) if which == "scipy" else nb.t.pdf(x, df, m, s)
    )


@pytest.mark.benchmark(group="t.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_t_cdf_speed(benchmark, which, n):
    x = np.linspace(-5, 5, n)
    df = np.linspace(1, 10, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(
        lambda: sc.t.cdf(x, df, m, s) if which == "scipy" else nb.t.cdf(x, df, m, s)
    )


@pytest.mark.benchmark(group="t.ppf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_t_ppf_speed(benchmark, which, n):
    x = np.linspace(0, 1, n)
    df = np.linspace(1, 10, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(
        lambda: sc.t.ppf(x, df, m, s) if which == "scipy" else nb.t.ppf(x, df, m, s)
    )
