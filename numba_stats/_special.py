# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from numba.extending import get_cython_function_address
import ctypes


def wrap(name, narg):
    addr = get_cython_function_address("scipy.special.cython_special", name)
    functype = ctypes.CFUNCTYPE(ctypes.c_double, *([ctypes.c_double] * narg))
    return functype(addr)


erfinv = wrap("erfinv", 1)
gammaincc = wrap("gammaincc", 2)
erf = wrap("__pyx_fuse_1erf", 1)
gammaln = wrap("gammaln", 1)
xlogy = wrap("__pyx_fuse_1xlogy", 2)
pdtr = wrap("pdtr", 2)
expm1 = wrap("__pyx_fuse_1expm1", 1)
log1p = wrap("__pyx_fuse_1log1p", 1)
