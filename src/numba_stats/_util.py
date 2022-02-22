import numba as nb
import functools
from typing import Callable


def _vectorize(narg, cache=True, **kwargs):
    def outer(func):

        signatures = [arg(*([arg] * narg)) for arg in (nb.float32, nb.float64)]

        wrapped = nb.vectorize(signatures, cache=cache, **kwargs)(func)
        functools.update_wrapper(wrapped, func)

        return wrapped

    return outer


def _jit(*args, cache=True, **kwargs):
    if len(args) == 1 and isinstance(args[0], Callable):
        return nb.njit(cache=cache, **kwargs)(args[0])

    def outer(func):
        return nb.njit(cache=cache, **kwargs)(func)

    return outer
