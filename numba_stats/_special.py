# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from numba.extending import get_cython_function_address
import ctypes


def make_fcn(name, narg):
    addr = get_cython_function_address("scipy.special.cython_special", name)
    functype = ctypes.CFUNCTYPE(ctypes.c_double, *([ctypes.c_double] * narg))
    return functype(addr)


erfinv = make_fcn("erfinv", 1)
gammaincc = make_fcn("gammaincc", 2)
erf = make_fcn("__pyx_fuse_1erf", 1)
gamma = make_fcn("__pyx_fuse_1gamma", 1)
