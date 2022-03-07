import numba as nb
import numpy as np
from numba.types import Array
from numba.core.errors import TypingError
from numba.extending import overload

_Floats = (nb.float32, nb.float64)


def _jit(arg, cache=True):
    if isinstance(arg, list):
        return nb.njit(arg, cache=cache, error_model="numpy")

    signatures = []
    for T in (nb.float32, nb.float64):
        if arg < 0:
            sig = T(*([T] * -arg))
        elif arg == 0:
            sig = T[:](T[:])
        else:
            sig = T[:](T[:], *([T] * arg))
        signatures.append(sig)

    return nb.njit(signatures, cache=cache, error_model="numpy")


def _wrap(fn):
    def outer(first, *rest):
        shape = np.shape(first)
        first = np.array(first).flatten()
        if first.dtype.kind != "f":
            first = first.astype(float)
        return fn(first, *rest).reshape(shape)

    return outer


@_jit(2)
def _trans(x, loc, scale):
    inv_scale = type(scale)(1) / scale
    return (x - loc) * inv_scale


def _type_check(first, *rest):
    if not (isinstance(first, Array) and first.dtype in _Floats):
        raise TypingError("first argument must be an array of floating point type")

    T = type(first.dtype)
    for i, tp in enumerate(rest):
        if not isinstance(tp, T):
            raise TypingError(f"argument {i+1} must be of type {tp}")


def _generate_wrappers(d):
    import inspect

    if "_wrap" not in d:
        d["_wrap"] = _wrap
    if "_type_check" not in d:
        d["_type_check"] = _type_check
    d["_overload"] = overload

    for fname in "pdf", "pmf", "logpdf", "logpmf", "cdf", "ppf", "density", "integral":
        impl = f"_{fname}"
        if impl not in d:
            continue
        fn = d[impl]
        args = inspect.signature(fn).parameters
        args = ", ".join([f"{x}" for x in args])
        code = f"""
def {fname}({args}):
    return _wrap({impl})({args})

{fname}.__doc__ = {impl}.__doc__

@_overload({fname})
def _ol_{fname}({args}):
    _type_check({args})
    return {impl}.__wrapped__
"""
        exec(code, d)
