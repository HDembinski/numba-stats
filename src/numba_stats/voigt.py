"""
Voigtian distribution.

This is the convolution of a Cauchy distribution with a normal distribution.

There is a closed form for the cdf, but the required hypergeometric function is not
implemented anywhere.

https://en.wikipedia.org/wiki/Voigt_profile

See Also
--------
scipy.special.voigt_profile: Equvialent in Scipy.
"""
from ._special import voigt_profile as _voigt
from ._util import _jit, _generate_wrappers, _prange, _to_array
import numpy as np

_doc_par = """
x : ArrayLike
    Random variate.
gamma : float
    The half-width at half-maximum of the Cauchy distribution part.
loc : float
    Location of the mode.
scale : float
    Standard deviation of the normal distribution.
"""


@_jit(3, cache=False)
def _pdf(x, gamma, loc, scale):
    x, shape = _to_array(x)
    r = np.empty_like(x)
    for i in _prange(len(x)):
        r[i] = _voigt(x[i] - loc, scale, gamma)
    return np.reshape(r, shape)


_generate_wrappers(globals())
