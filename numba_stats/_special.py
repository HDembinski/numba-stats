# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from numba.extending import get_cython_function_address
import ctypes
import scipy.special.cython_special as cysp


def get(name, narg):
    pyx_fuse_name = f"__pyx_fuse_1{name}"
    if pyx_fuse_name in cysp.__pyx_capi__:
        name = pyx_fuse_name
    addr = get_cython_function_address("scipy.special.cython_special", name)
    functype = ctypes.CFUNCTYPE(ctypes.c_double, *([ctypes.c_double] * narg))
    return functype(addr)


erfinv = get("erfinv", 1)
gammaincc = get("gammaincc", 2)
erf = get("erf", 1)
gammaln = get("gammaln", 1)
xlogy = get("xlogy", 2)
pdtr = get("pdtr", 2)
expm1 = get("expm1", 1)
log1p = get("log1p", 1)
stdtr = get("stdtr", 2)
stdtrit = get("stdtrit", 2)
