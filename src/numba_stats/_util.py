import numba as nb
import functools
from typing import Callable


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


def _jit(*args, **kwargs):
    if "error_model" not in kwargs:
        kwargs["error_model"] = "numpy"

    if len(args) == 1 and isinstance(args[0], Callable):
        return nb.njit(**kwargs)(args[0])

    def outer(func):
        return nb.njit(**kwargs)(func)

    return outer
