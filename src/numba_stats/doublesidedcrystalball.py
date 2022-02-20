import numba as nb
import numpy as np
from math import erf as _erf


@nb.njit(cache=True)
def _pdf(z, betal, ml, betar, mr):
    assert betal > 0
    assert betar > 0
    assert ml > 1
    assert mr > 1

    exp_betal = np.exp(-0.5 * betal ** 2)
    exp_betar = np.exp(-0.5 * betar ** 2)

    c = ml / (betal * (ml - 1.0)) * exp_betal
    d = np.sqrt(0.5 * np.pi) * (
        _erf(betar * np.sqrt(0.5)) - _erf(-betal * np.sqrt(0.5))
    )
    e = mr / (betar * (mr - 1.0)) * exp_betar
    n = 1.0 / (c + d + e)

    if z <= -betal:
        a = (ml / betal) ** ml * exp_betal
        b = ml / betal - betal
        return n * a * (b - z) ** -ml
    elif z >= betar:
        a = (mr / betar) ** mr * exp_betar
        b = mr / betar - betar
        return n * a * (b + z) ** -mr
    return n * np.exp(-0.5 * z ** 2)


@nb.njit(cache=True)
def _cdf(z, betal, ml, betar, mr):
    exp_betal = np.exp(-0.5 * betal ** 2)
    exp_betar = np.exp(-0.5 * betar ** 2)

    c = ml / (betal * (ml - 1.0)) * exp_betal
    d = np.sqrt(0.5 * np.pi) * (
        _erf(betar * np.sqrt(0.5)) - _erf(-betal * np.sqrt(0.5))
    )
    e = mr / (betar * (mr - 1.0)) * exp_betar
    n = 1.0 / (c + d + e)

    if z <= -betal:
        return n * (
            (ml / betal) ** ml
            * exp_betal
            * (ml / betal - betal - z) ** (1.0 - ml)
            / (ml - 1.0)
        )
    elif z >= betar:
        return n * (
            (ml / betal) * exp_betal / (ml - 1.0)
            + np.sqrt(0.5 * np.pi)
            * (_erf(betar * np.sqrt(0.5)) - _erf(-betal * np.sqrt(0.5)))
            + (mr / betar) ** mr
            * exp_betar
            * (mr / betar - betar + z) ** (1.0 - mr)
            / (1.0 - mr)
            + (mr / betar) * exp_betar / (mr - 1)
        )
    return n * (
        (ml / betal) * exp_betal / (ml - 1.0)
        + np.sqrt(0.5 * np.pi) * (_erf(z * np.sqrt(0.5)) - _erf(-betal * np.sqrt(0.5)))
    )


_signatures = [
    nb.float32(
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32,
    ),
    nb.float64(
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, betal, ml, betar, mr, loc, scale):
    z = (x - loc) / scale
    return _pdf(z, betal, ml, betar, mr) / scale


@nb.vectorize(_signatures, cache=True)
def cdf(x, betal, ml, betar, mr, loc, scale):
    z = (x - loc) / scale
    return _cdf(z, betal, ml, betar, mr)
