import numba as nb
import functools
import numpy as np


def _vectorize(narg, cache=True, **kwargs):
    def outer(func):
        if "cache" not in kwargs:
            kwargs["cache"] = cache
        # if "error_model" not in kwargs:
        #     kwargs["error_model"] = "numpy"

        signatures = [arg(*([arg] * narg)) for arg in (nb.float32, nb.float64)]

        wrapped = nb.vectorize(signatures, **kwargs)(func)
        # this does not help with the docs, unfortunately
        functools.update_wrapper(wrapped, func)

        return wrapped

    return outer


def _jit(narg, cache=True, **kwargs):
    if "error_model" not in kwargs:
        kwargs["error_model"] = "numpy"

    if "cache" not in kwargs:
        kwargs["cache"] = cache

    return nb.njit(**kwargs)

    signatures = []
    for arg in (nb.float32, nb.float64):
        if narg < 0:
            sig = arg(*([arg] * -narg))
        elif narg == 0:
            sig = arg[:](arg[:])
        else:
            sig = arg[:](arg[:], *([arg] * narg))
        signatures.append(sig)

    return nb.njit(signatures, **kwargs)


def _cast(x):
    x = np.atleast_1d(x)
    if x.dtype.kind != "f":
        return x.astype(float)
    return x


@_jit(2)
def _trans(x, loc, scale):
    inv_scale = 1 / scale
    return (x - loc) * inv_scale


def _type_check(fn, *types):
    import inspect
    from numba.core.errors import TypingError
    from numba.types import Array, Number

    signature = inspect.signature(fn)
    for i, (tp, par) in enumerate(zip(types, signature.parameters)):
        if i == 0:
            if not isinstance(tp, Array):
                raise TypingError(f"{par} must be an array")
        else:
            if not isinstance(tp, Number):
                raise TypingError(f"{par} must be number")
