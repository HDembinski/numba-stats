# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from numba.extending import get_cython_function_address
from numba.types import WrapperAddressProtocol, float64


def get(name, signature):
    # create new function object with correct signature that numba can call by extracting
    addr = get_cython_function_address("scipy.special.cython_special", name)

    # dynamically create type that inherits from WrapperAddressProtocol
    cls = type(
        name,
        (WrapperAddressProtocol,),
        {"__wrapper_address__": lambda self: addr, "signature": lambda self: signature},
    )
    return cls()


# unary functions (double)
ndtri = get("ndtri", float64(float64))

# binary functions (double)
gammaincc = get("gammaincc", float64(float64, float64))
stdtrit = get("stdtrit", float64(float64, float64))
stdtr = get("stdtr", float64(float64, float64))
betainc = get("betainc", float64(float64, float64, float64))

# n-ary functions (double)
voigt_profile = get("voigt_profile", float64(float64, float64, float64))
