import numba as nb
import numpy as np

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
    def outer(arg, *rest):
        shape = np.shape(arg)
        arg = np.atleast_1d(arg).flatten()
        if arg.dtype.kind != "f":
            arg = arg.astype(float)
        return fn(arg, *rest).reshape(shape)

    return outer


def _cast(x):
    x = np.atleast_1d(x)
    if x.dtype.kind != "f":
        return x.astype(float)
    return x


@_jit(2)
def _trans(x, loc, scale):
    inv_scale = type(scale)(1) / scale
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
