# numba-stats

![](https://img.shields.io/pypi/v/numba-stats.svg)

We provide numba-accelerated implementations of statistical functions for common probability distributions

* Uniform
* (Truncated) Normal
* Log-normal
* Poisson
* (Truncated) Exponential
* Student's t
* Voigtian
* Crystal Ball
* Generalised double-sided Crystal Ball
* Tsallis-Hagedorn, a model for the minimum bias pT distribution
* Q-Gaussian
* Bernstein density (not normalised to unity, use this in extended likelihood fits)

with more to come. The speed gains are huge, up to a factor of 100 compared to `scipy`. Benchmarks are included in the repository and are run by `pytest`.

## Documentation (or lack of)

Because of a technical limitation of Numba, this project is poorly documented. Functions with equivalents in `scipy.stats` follow the Scipy calling conventions exactly. These conventions are sometimes a bit unusual, for example, in case of the exponential, the log-normal or the uniform distribution. See the SciPy docs for details.

Please look into the source code for documentation of the other functions.

Technical note: `pydoc numba_stats` does not show anything useful, because `numba.vectorize` creates instances of a class `DUFunc`. The wrapped functions show up as objects of that class and `help()` shows the generic documentation of that class instead of the documentation for the instances.

## Contributions

**You can help with adding more distributions, patches are very welcome.** Implementing a probability distribution is easy. You need to write it in simple Python that `numba` can understand. Special functions from `scipy.special` can be used after some wrapping, see submodule `numba_stats._special.py` how it is done.

## Plans for version 1.0

Version v1.0 will introduce breaking changes to the API. Users are recommended to update their code.
```
# before v0.8
from numba_stats import norm_pdf
from numba_stats.stats import norm_cdf

dp = norm_pdf(1, 2, 3)
p = norm_cdf(1, 2, 3)

# recommended since v0.8
from numba_stats import norm

dp = norm.pdf(1, 2, 3)
p = norm.cdf(1, 2, 3)
```
This is nicer code, but more importantly, this is necessary to battle the increasing startup times of `numba-stats`. Now you only pay the compilation cost for the distribution that you actually import. The `stats` submodule will be removed. To keep old code running, please pin your numba_stats to version `<1`.

## numba-stats and numba-scipy

[numba-scipy](https://github.com/numba/numba-scipy) is the official package and repository for fast numba-accelerated scipy functions, are we reinventing the wheel?

Ideally, the functionality in this package should be in `numba-scipy` and we hope that eventually this will be case. In this package, we don't offer overloads for scipy functions and classes like `numba-scipy` does. This simplifies the implementation dramatically. `numba-stats` is intended as a temporary solution until fast statistical functions are included in `numba-scipy`. `numba-stats` currently does not depend on `numba-scipy`, only on `numba` and `scipy`.
