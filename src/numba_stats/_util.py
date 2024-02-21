"""Utilities for code and docs generation to reduce boilerplate code."""

import math
import numba as nb
import numpy as np
from numba.types import Array
from numba.core.errors import TypingError
from numba.extending import overload
from numba import prange as _prange  # noqa
import os

_Floats = (nb.float32, nb.float64)


def _readonly_carray(T):
    return Array(T, 1, "A", readonly=True)


def _jit(arg, cache=True):
    """
    Wrap numba.njit to reduce boilerplate code.

    We want to build jitted functions with explicit signatures to restrict the argument
    types which are used in the implemnetation to float32 or float64. We also want to
    pass specific options consistently: error_model='numpy' and inline='always'. The
    latter is important to profit from auto-parallelization of surrounding code.

    Parameters
    ----------
    arg : int
        Number of arguments. If negative, all arguments of this function are scalars
        and -arg is the number of arguments. If positive, the first argument is
        an array, the others are scalars and arg is the number of scalar arguments.
    """
    if isinstance(arg, list):
        signatures = arg
    else:
        signatures = []
        for T in (nb.float32, nb.float64):
            if arg < 0:
                sig = T(*([T] * -arg))
            else:
                sig = T[:](_readonly_carray(T), *[T for _ in range(arg)])
            signatures.append(sig)
    return nb.njit(signatures, cache=cache, inline="always", error_model="numpy")


def _rvs_jit(arg, cache=True):
    signatures = []
    T = nb.float64  # nb.float32 cannot be supported
    # extra args at the end are for size and random_state
    sig = T[:](*[T for _ in range(arg)], nb.uint64, nb.optional(nb.uint64))
    signatures.append(sig)
    return nb.njit(signatures, cache=cache, inline="always", error_model="numpy")


@nb.njit(cache=True)
def _seed(seed):
    if seed is None:
        with nb.objmode(seed="optional(uint8)"):
            seed = np.frombuffer(os.urandom(8), dtype=np.uint64)[0]
    np.random.seed(seed)


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


@nb.njit(cache=True, inline="always", error_model="numpy")
def _erf_inplace(x):
    for i in _prange(len(x)):
        x[i] = math.erf(x[i])


@nb.njit(cache=True, inline="always", error_model="numpy")
def _erfc_inplace(x):
    for i in _prange(len(x)):
        x[i] = math.erfc(x[i])


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

    doc_par = d["_doc_par"].strip() if "_doc_par" in d else None

    for fname in (
        "pdf",
        "pmf",
        "logpdf",
        "logpmf",
        "cdf",
        "ppf",
        "density",
        "integral",
        "rvs",
    ):
        impl = f"_{fname}"
        if impl not in d:
            continue
        fn = d[impl]
        args = inspect.signature(fn).parameters
        args = ", ".join([f"{x}" for x in args])
        doc_title = {
            "density": "Return density.",
            "integral": "Return integrated density.",
            "logpdf": "Return log of probability density.",
            "logpmf": "Return log of probability mass.",
            "pmf": "Return probability mass.",
            "pdf": "Return probability density.",
            "cdf": "Return cumulative probability.",
            "ppf": "Return quantile for given probability.",
            "rvs": "Return random samples from distribution.",
        }.get(fname, None)
        if fname == "ppf":
            before_par = """\
x: ArrayLike
    Probability. Must be between 0 and 1.
"""
        elif fname == "rvs":
            before_par = ""
        else:
            before_par = """\
x: ArrayLike
    Random variate.
"""
        if fname == "rvs":
            after_par = """
size : int
    Number of random variates.
random_state : int or None
    Seed of the random number generator. Default is None, which uses a random seed."""
        else:
            after_par = ""

        if fname == "rvs":
            code = f"""
def {fname}({args}):
    return {impl}({args})

@_overload({fname}, inline="always")
def _ol_{fname}({args}):
    return {impl}
"""
        else:
            code = f"""
def {fname}({args}):
    return _wrap({impl})({args})

@_overload({fname}, inline="always")
def _ol_{fname}({args}):
    _type_check({args})
    return {impl}.__wrapped__
"""

        if doc_par is None:
            code += f"""
{fname}.__doc__ = {impl}.__doc__
"""
        else:
            assert doc_title is not None
            code += f"""
{fname}.__doc__ = \"\"\"
{doc_title}

Parameters
----------
{before_par}{doc_par}{after_par}

Returns
-------
Array-like
    Function evaluated at the x points.
\"\"\"
"""
        exec(code, d)
