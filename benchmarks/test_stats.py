import scipy.stats as sc
import numpy as np
import pytest


@pytest.mark.benchmark(group="norm.pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_pdf_speed(benchmark, which, n):
    from numba_stats import norm

    x = np.linspace(-5, 5, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.norm.pdf(x, m, s) if which == "scipy" else norm.pdf(x, m, s))


@pytest.mark.benchmark(group="norm.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_cdf_speed(benchmark, which, n):
    from numba_stats import norm

    x = np.linspace(-5, 5, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.norm.cdf(x, m, s) if which == "scipy" else norm.cdf(x, m, s))


@pytest.mark.benchmark(group="norm.ppf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_norm_ppf_speed(benchmark, which, n):
    from numba_stats import norm

    x = np.linspace(0, 1, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.norm.ppf(x, m, s) if which == "scipy" else norm.ppf(x, m, s))


@pytest.mark.benchmark(group="poisson.pmf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_poisson_pmf_speed(benchmark, which, n):
    from numba_stats import poisson

    k = np.arange(0, n)
    m = np.linspace(0, 10, n)
    benchmark(lambda: sc.poisson.pmf(k, m) if which == "scipy" else poisson.pmf(k, m))


@pytest.mark.benchmark(group="poisson.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_poisson_cdf_speed(benchmark, which, n):
    from numba_stats import poisson

    k = np.arange(0, n)
    m = np.linspace(0, 10, n)
    benchmark(lambda: sc.poisson.cdf(k, m) if which == "scipy" else poisson.cdf(k, m))


@pytest.mark.benchmark(group="expon.pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_expon_pdf_speed(benchmark, which, n):
    from numba_stats import expon

    x = np.linspace(0, 10, n)
    m = np.linspace(0, 10, n)
    s = np.linspace(1, 10, n)
    benchmark(lambda: sc.expon.pdf(x, m, s) if which == "scipy" else expon.pdf(x, m, s))


@pytest.mark.benchmark(group="expon.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_expon_cdf_speed(benchmark, which, n):
    from numba_stats import expon

    x = np.linspace(0, 10, n)
    m = np.linspace(0, 10, n)
    s = np.linspace(1, 10, n)
    benchmark(lambda: sc.expon.cdf(x, m, s) if which == "scipy" else expon.cdf(x, m, s))


@pytest.mark.benchmark(group="t.pdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_t_pdf_speed(benchmark, which, n):
    from numba_stats import t

    x = np.linspace(-5, 5, n)
    df = np.linspace(1, 10, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.t.pdf(x, df, m, s) if which == "scipy" else t.pdf(x, df, m, s))


@pytest.mark.benchmark(group="t.cdf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_t_cdf_speed(benchmark, which, n):
    from numba_stats import t

    x = np.linspace(-5, 5, n)
    df = np.linspace(1, 10, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.t.cdf(x, df, m, s) if which == "scipy" else t.cdf(x, df, m, s))


@pytest.mark.benchmark(group="t.ppf")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_t_ppf_speed(benchmark, which, n):
    from numba_stats import t

    x = np.linspace(0, 1, n)
    df = np.linspace(1, 10, n)
    m = np.linspace(-1, 1, n)
    s = np.linspace(0.1, 1, n)
    benchmark(lambda: sc.t.ppf(x, df, m, s) if which == "scipy" else t.ppf(x, df, m, s))


@pytest.mark.benchmark(group="bernstein.density")
@pytest.mark.parametrize("which", ("scipy", "ours"))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
def test_bernstein_density_speed(benchmark, which, n):
    from numba_stats import bernstein
    from scipy.interpolate import BPoly

    x = np.linspace(0, 1, n)
    beta = np.arange(1, 4, dtype=float)

    benchmark(
        lambda: BPoly(np.array(beta)[:, np.newaxis], [x[0], x[-1]])(x)
        if which == "scipy"
        else bernstein.density(x, beta, x[0], x[-1])
    )
