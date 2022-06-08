"""
Poisson distribution.

See Also
--------
scipy.stats.poisson: Scipy equivalent.
"""

import numpy as np
from ._special import gammaincc as _gammaincc
from math import lgamma as _lgamma
from ._util import _jit, _generate_wrappers, _prange
from .special import erfinv as _erfinv
from math import erfc as _erfc

_doc_par = """
x : ArrayLike
    Random variate.
mu : float
    Expected value.
"""


@_jit(1)
def _logpmf(k, mu):
    T = type(mu)
    r = np.empty(len(k), T)
    for i in _prange(len(r)):
        if mu == 0:
            r[i] = 0.0 if k[i] == 0 else -np.inf
        else:
            r[i] = k[i] * np.log(mu) - _lgamma(k[i] + T(1)) - mu
    return r


@_jit(1)
def _pmf(k, mu):
    return np.exp(_logpmf(k, mu))


@_jit(1, cache=False)
def _cdf(k, mu):
    T = type(mu)
    r = np.empty(len(k), T)
    for i in _prange(len(r)):
        r[i] = _gammaincc(k[i] + T(1), mu)
    return r


@_jit(-2)
def _ppf1(u, lam):
    v = 1.0 - u
    if u == 0.0:
        return 0.0
    if v == 0.0:
        return np.inf
    if not (u > 0.0 and v > 0.0):
        return np.nan

    x = 0.0
    lam_inv = 1 / lam
    sq2 = np.sqrt(2)

    if lam > 4.0:
        w = sq2 * _erfinv(2 * np.minimum(u, v) - 1)
        if u > v:
            w = -w

        if np.abs(w) < 3.0:
            lr = np.sqrt(lam)
            s = lr * w + (1.0 / 3.0 + (1.0 / 6.0) * w * w) * (1.0 - w / (12.0 * lr))

            d = 1.0 / 160.0
            d = (1.0 / 80.0) + d * (w * w)
            d = (1.0 / 40.0) + d * (w * w)
            d = d * lam_inv

            s = lam + (s + d)
        else:
            s = w / np.sqrt(lam)
            r = 1.0 + s
            if r < 0.1:
                r = 0.1
            r2 = 0.0

            while np.abs(r - r2) > 1e-8:
                t = np.log(r)
                r2 = r
                s2 = np.sqrt(2.0 * ((1.0 - r) + r * t))
                if r < 1.0:
                    s2 = -s2
                r = r2 - (s2 - s) * s2 / t
                if r < 0.1 * r2:
                    r = 0.1 * r2

            t = np.log(r)
            s = (
                lam * r
                + np.log(np.sqrt(2.0 * r * ((1.0 - r) + r * t)) / np.abs(r - 1.0)) / t
            )
            s = s - 0.0218 / (s + 0.065 * lam)
            d = 0.01 / s
            s = s + d

        x = np.floor(s)

        if 10.0 < s < x + 2.0 * d:
            xi = 1.0 / x
            eta = x * lam_inv
            eta = np.sqrt(2.0 * (1.0 - eta + eta * np.log(eta)) / eta)
            if x > lam:
                eta = -eta

            b1 = 8.0995211567045583e-16
            s = b1
            b0 = -1.9752288294349411e-15
            s = b0 + s * eta
            b1 = -5.1391118342426808e-16 + 25.0 * b1 * xi
            s = b1 + s * eta
            b0 = 2.8534893807047458e-14 + 24.0 * b0 * xi
            s = b0 + s * eta
            b1 = -1.3923887224181616e-13 + 23.0 * b1 * xi
            s = b1 + s * eta
            b0 = 3.3717632624009806e-13 + 22.0 * b0 * xi
            s = b0 + s * eta
            b1 = 1.1004392031956284e-13 + 21.0 * b1 * xi
            s = b1 + s * eta
            b0 = -5.0276692801141763e-12 + 20.0 * b0 * xi
            s = b0 + s * eta
            b1 = 2.4361948020667402e-11 + 19.0 * b1 * xi
            s = b1 + s * eta
            b0 = -5.8307721325504166e-11 + 18.0 * b0 * xi
            s = b0 + s * eta
            b1 = -2.5514193994946487e-11 + 17.0 * b1 * xi
            s = b1 + s * eta
            b0 = 9.1476995822367933e-10 + 16.0 * b0 * xi
            s = b0 + s * eta
            b1 = -4.3820360184533521e-09 + 15.0 * b1 * xi
            s = b1 + s * eta
            b0 = 1.0261809784240299e-08 + 14.0 * b0 * xi
            s = b0 + s * eta
            b1 = 6.7078535434015332e-09 + 13.0 * b1 * xi
            s = b1 + s * eta
            b0 = -1.7665952736826086e-07 + 12.0 * b0 * xi
            s = b0 + s * eta
            b1 = 8.2967113409530833e-07 + 11.0 * b1 * xi
            s = b1 + s * eta
            b0 = -1.8540622107151585e-06 + 10.0 * b0 * xi
            s = b0 + s * eta
            b1 = -2.1854485106799979e-06 + 9.0 * b1 * xi
            s = b1 + s * eta
            b0 = 3.9192631785224383e-05 + 8.0 * b0 * xi
            s = b0 + s * eta
            b1 = -0.00017875514403292177 + 7.0 * b1 * xi
            s = b1 + s * eta
            b0 = 0.00035273368606701921 + 6.0 * b0 * xi
            s = b0 + s * eta
            b1 = 0.0011574074074074078 + 5.0 * b1 * xi
            s = b1 + s * eta
            b0 = -0.014814814814814815 + 4.0 * b0 * xi
            s = b0 + s * eta
            b1 = 0.083333333333333329 + 3.0 * b1 * xi
            s = b1 + s * eta
            b0 = -0.33333333333333331 + 2.0 * b0 * xi
            s = b0 + s * eta
            s = s / (1.0 + b1 * xi)

            # there is something strange in this formula
            s = s * np.exp(-0.5 * x * eta * eta) / np.sqrt(2.0 * np.pi * x)
            if x < lam:
                s += 0.5 * _erfc(eta * np.sqrt(0.5 * x))
                # s += 1.0 - norm.cdf(eta * np.sqrt(x), 0.0, 1.0)
                if s > u:
                    x -= 1.0
            else:
                s -= 0.5 * _erfc(-eta * np.sqrt(0.5 * x))
                # s -= 1.0 - norm.cdf(-eta * np.sqrt(x), 0.0, 1.0)
                if s > -v:
                    x -= 1.0
        else:
            xi = 1.0 / x
            s = -691.0 / 360360.0
            s = 1.0 / 1188.0 + s * xi * xi
            s = -1.0 / 1680.0 + s * xi * xi
            s = 1.0 / 1260.0 + s * xi * xi
            s = -1.0 / 360.0 + s * xi * xi
            s = 1.0 / 12.0 + s * xi * xi
            s = s * xi
            s = (x - lam) - x * np.log(x * lam_inv) - s

            if x < lam:
                t = np.exp(-0.5 * s)
                s = 1.0 - t * (u * t) * np.sqrt(2.0 * 3.141592653589793 * xi) * lam
                t = 1.0
                xi = x
                for i in range(1, 50):
                    xi -= 1.0
                    t *= xi * lam_inv
                    s += t

                if s > 0.0:
                    x -= 1.0
            else:
                t = np.exp(-0.5 * s)
                s = 1.0 - t * (v * t) * np.sqrt(2.0 * 3.141592653589793 * x)
                xi = x
                for i in range(1, 50):
                    xi += 1.0
                    s = s * xi * lam_inv + 1.0

                if s < 0.0:
                    x -= 1.0

    if x < 10.0:
        x = 0.0
        t = np.exp(0.5 * lam)
        d = 0.0
        if u > 0.5:
            d = t * (1e-13 * t)
        s = 1.0 - t * (u * t) + d

        while s < 0.0:
            x += 1.0
            t = x * lam_inv
            d = t * d
            s = t * s + 1.0

        if s < 2.0 * d:
            d = 1e13 * d
            t = 1e17 * d
            d = v * d

            while d < t:
                x += 1.0
                d *= x * lam_inv

            s = d
            t = 1.0
            while s > 0.0:
                t *= x * lam_inv
                s -= t
                x -= 1.0

    return x


@_jit(1)
def _ppf(p, mu):
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = _ppf1(p[i], mu)
    return r


_generate_wrappers(globals())
