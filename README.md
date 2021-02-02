# numba-stats

We provide numba-accelerated implementations of statistical functions for common probability distributions

* normal
* poisson
* exponential

with more to come. The speed gains are huge, up to a factor of 100 compared to `scipy`.

**You can help with adding more distributions, patches are very welcome.** Implementing a probability distribution is easy. You need to write it in simple Python that numba can understand. Special functions from `scipy.special` can be used after some custom wrapping, see submodule `numba_stats._special.py` how it is done.

Because of limited manpower, this project is barely documented. Use `pydoc numba_stats`
