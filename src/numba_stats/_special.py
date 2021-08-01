# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from numba.extending import get_cython_function_address
from numba.types import WrapperAddressProtocol, float64
import scipy.special.cython_special as cysp
import numba as nb
import numpy as np


def get(name, signature):
    # create new function object with correct signature that numba can call by extracting
    # function pointer from scipy.special.cython_special; uses scipy/cython internals
    index = 1 if signature.return_type is float64 else 0
    pyx_fuse_name = f"__pyx_fuse_{index}{name}"
    if pyx_fuse_name in cysp.__pyx_capi__:
        name = pyx_fuse_name
    addr = get_cython_function_address("scipy.special.cython_special", name)

    # dynamically create type that inherits from WrapperAddressProtocol
    cls = type(
        name,
        (WrapperAddressProtocol,),
        {"__wrapper_address__": lambda self: addr, "signature": lambda self: signature},
    )
    return cls()


# unary functions (double)
erfinv = get("erfinv", float64(float64))

# binary functions (double)
gammaincc = get("gammaincc", float64(float64, float64))
stdtrit = get("stdtrit", float64(float64, float64))
hyp2f1 = get("hyp2f1", float64(float64, float64, float64, float64))
betainc = get("betainc", float64(float64, float64, float64))


@nb.njit
def stdtr(nu, t):
    # supports real values for nu, while scipy.special.stdtr current does not
    if nu <= 0:
        return np.nan

    if t == 0:
        return 0.5

    x = t if t < 0 else -t
    z = nu / (nu + x * x)
    p = 0.5 * betainc(0.5 * nu, 0.5, z)

    if t < 0:
        return p
    return 1 - p


# n-ary functions (double)
voigt_profile = get("voigt_profile", float64(float64, float64, float64))
