import numba as nb
from ._special import voigt_profile

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures)
def pdf(x, gamma, mu, sigma):
    """
    Return probability density of Voigtian distribution.
    """
    return voigt_profile(x - mu, gamma, sigma)
