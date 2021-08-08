import numpy as np
import numba as nb


_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.njit(_signatures, cache=True)
def _de_castlejau(z, beta, res):
    # De Casteljau algorithm, numerically stable
    n = len(beta)
    if n == 0:
        res[:] = np.nan
    else:
        betai = np.empty_like(beta)
        for iz, zi in enumerate(z):
            azi = 1.0 - zi
            betai[:] = beta
            for j in range(1, n):
                for k in range(n - j):
                    betai[k] = betai[k] * azi + betai[k + 1] * zi
            res[iz] = betai[0]
    return res


_signatures = [
    nb.float32[:](nb.float32[:]),
    nb.float64[:](nb.float64[:]),
]


@nb.njit(_signatures, cache=True)
def _beta_int(beta):
    n = len(beta)
    r = np.zeros(n + 1, dtype=beta.dtype)
    for j in range(1, n + 1):
        for k in range(j):
            r[j] += beta[k]
    r *= 1.0 / n
    return r


@nb.njit(cache=True)
def _prepare_z_beta(x, xmin, xmax, beta):
    inverse_scale = 1 / (xmax - xmin)
    z = x.copy()
    z -= xmin
    z *= inverse_scale
    # beta = beta.copy()
    # inverse_scale /= len(beta) + 1
    # beta *= inverse_scale
    return z, beta


def _prepare_array(x):
    x = np.atleast_1d(x)
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    return x


_signatures = [
    (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]),
    (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
]


@nb.guvectorize(_signatures, "(),(n),(),()->()", cache=True)
def scaled_pdf(x, beta, xmin, xmax, res):
    z, beta = _prepare_z_beta(x, xmin, xmax, beta)
    _de_castlejau(z, beta, res)


@nb.guvectorize(_signatures, "(),(n),(),()->()", cache=True)
def scaled_cdf(x, beta, xmin, xmax, res):
    z, beta = _prepare_z_beta(x, xmin, xmax, beta)
    beta = _beta_int(beta)
    _de_castlejau(z, beta, res)


@nb.extending.overload(scaled_pdf)
def bernstein_scaled_pdf_ol(x, beta, xmin, xmax):
    from numba.core.errors import TypingError
    from numba.types import Array, Float

    if not isinstance(x, Array):
        raise TypingError("x must be a Numpy array")
    if not isinstance(beta, Array):
        raise TypingError("beta must be a Numpy array")
    if not isinstance(xmin, Float):
        raise TypingError("xmin must be float")
    if not isinstance(xmax, Float):
        raise TypingError("xmax must be float")

    def impl(x, beta, xmin, xmax):
        z, beta = _prepare_z_beta(x, xmin, xmax, beta)
        res = np.empty_like(z)
        _de_castlejau(z, beta, res)
        return res

    return impl


@nb.extending.overload(scaled_cdf)
def bernstein_scaled_cdf_ol(x, beta, xmin, xmax):
    from numba.core.errors import TypingError
    from numba.types import Array, Float

    if not isinstance(x, Array):
        raise TypingError("x must be a Numpy array")
    if not isinstance(beta, Array):
        raise TypingError("beta must be a Numpy array")
    if not isinstance(xmin, Float):
        raise TypingError("xmin must be float")
    if not isinstance(xmax, Float):
        raise TypingError("xmax must be float")

    def impl(x, beta, xmin, xmax):
        z, beta = _prepare_z_beta(x, xmin, xmax, beta)
        beta = _beta_int(beta)
        res = np.empty_like(z)
        _de_castlejau(z, beta, res)
        return res

    return impl


density = scaled_pdf
