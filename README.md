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

## Usage

Each distribution is implemented in a submodule. Import the submodule that you need.
```py
from numba_stats import norm
import numpy as np

x = np.linspace(-10, 10)
mu = 2
sigma = 3

dp = norm.pdf(x, mu, sigma)
p = norm.cdf(x, mu, sigma)
```
The functions are fully vectorised, which means that `mu` and `sigma` can be vectors, too, although this is not usually needed. In the best case, the following functions are implemented
* `logpdf`
* `pdf`
* `cdf`
* `ppf`
`logpdf` is only implemented if it is more efficient and accurate compared to computing `log(dist.pdf(...))`. `cdf` and `ppf` are missing for some distributions (e.g. `voigt`), if there is no known way to compute them accurately.

## Documentation (or lack of)

Because of a technical limitation of Numba, this project is poorly documented. Functions with equivalents in `scipy.stats` follow the Scipy calling conventions exactly. These conventions are sometimes a bit unusual, for example, in case of the exponential, the log-normal or the uniform distribution. See the SciPy docs for details.

Please look into the source code for documentation of the other functions.

Technical note: `pydoc numba_stats` does not show anything useful, because `numba.vectorize` creates instances of a class `DUFunc`. The wrapped functions show up as objects of that class and `help()` shows the generic documentation of that class instead of the documentation for the instances.

## Contributions

**You can help with adding more distributions, patches are very welcome.** Implementing a probability distribution is easy. You need to write it in simple Python that `numba` can understand. Special functions from `scipy.special` can be used after some wrapping, see submodule `numba_stats._special.py` how it is done.

## numba-stats and numba-scipy

[numba-scipy](https://github.com/numba/numba-scipy) is the official package and repository for fast numba-accelerated scipy functions, are we reinventing the wheel?

Ideally, the functionality in this package should be in `numba-scipy` and we hope that eventually this will be case. In this package, we don't offer overloads for scipy functions and classes like `numba-scipy` does. This simplifies the implementation dramatically. `numba-stats` is intended as a temporary solution until fast statistical functions are included in `numba-scipy`. `numba-stats` currently does not depend on `numba-scipy`, only on `numba` and `scipy`.
