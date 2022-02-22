from ._special import voigt_profile as _voigt
from ._util import _vectorize


@_vectorize(4, cache=False)
def pdf(x, gamma, loc, scale):
    """
    Return probability density of Voigtian distribution.
    """
    return _voigt(x - loc, scale, gamma)
