import numba as nb
import numpy as np
from numba.types import Array, Float, Integer
from numba.core.errors import TypingError
from numba.extending import overload
from numba import prange as _prange  # noqa

_Floats = (nb.float32, nb.float64)


def _to_array(a):
    return np.ravel(a), a.shape


@overload(_to_array, inline="always")
def _(a):
    if isinstance(a, (Float, Integer)):

        def _(a):
            return np.array([a]), ()

        return _

    if isinstance(a, Array):
        if a.ndim == 0:

            def _(a):
                return a.reshape((1,)), ()

        elif a.ndim == 1:

            def _(a):
                return a, a.shape

        else:

            def _(a):
                return a.ravel(), a.shape

        return _


def _readonly_carray(T):
    return Array(T, 1, "A", readonly=True)


def _jit(arg, scalar=True, cache=True):
    """
    Wrapper for numba.njit to reduce boilerplate code.

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
        return nb.njit(arg, cache=cache, inline="always", error_model="numpy")

    signatures = []
    for T in (nb.float32, nb.float64):
        if arg < 0:
            sig = T(*([T] * -arg))
            signatures.append(sig)
        else:
            sig = T[:](_readonly_carray(T), *[T for _ in range(arg)])
            signatures.append(sig)
            if scalar:
                sig = T(T, *[T for _ in range(arg)])
                signatures.append(sig)

    return nb.njit(signatures, cache=cache, inline="always", error_model="numpy")


def _wrap(fn):
    def outer(first, *rest):
        if not isinstance(first, np.ndarray):
            first = np.ndarray(first)
        if first.dtype.kind != "f":
            first = first.astype(float)
        return fn(first, *rest)

    return outer


@_jit(2, scalar=False)
def _trans(x, loc, scale):
    inv_scale = type(scale)(1) / scale
    return (x - loc) * inv_scale


def _type_check(first, *rest):
    if isinstance(first, (Float, Integer)):
        return

    if not (isinstance(first, Array) and first.dtype in _Floats):
        raise TypingError(
            "first argument must be a number or an array of floating point type"
        )

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

    for fname in "pdf", "pmf", "logpdf", "logpmf", "cdf", "ppf", "density", "integral":
        impl = f"_{fname}"
        if impl not in d:
            continue
        fn = d[impl]
        args = inspect.signature(fn).parameters
        args = ", ".join([f"{x}" for x in args])
        doc_title = {
            "density": "Return density.",
            "logpdf": "Return log of probability density.",
            "logpmf": "Return log of probability mass.",
            "pmf": "Return probability mass.",
            "pdf": "Return probability density.",
            "cdf": "Return cumulative probability.",
            "ppf": "Return quantile for given probability.",
        }.get(fname, None)

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
{doc_par}

Returns
-------
Array-like
    Function evaluated at the x points.
\"\"\"
"""
        exec(code, d)
