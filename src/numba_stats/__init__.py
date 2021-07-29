from ._version import version as __version__  # noqa


def __getattr__(key):
    # Temporary hack to maintain backward compatibility
    import importlib
    import warnings
    from numpy import VisibleDeprecationWarning

    try:
        dist, fn = key.split("_")
        if fn not in ("pdf", "cdf", "ppf", "pmf"):
            raise AttributeError

        warnings.warn(
            f"""Imports of the form `from numba_stats import {key}` will be removed in v1.0
Please import distributions like this:

from numba_stats import {dist}

{dist}.{fn}(...)
""",
            VisibleDeprecationWarning,
            1,
        )
        dist = importlib.import_module(f"numba_stats.{dist}")
        return getattr(dist, fn)
    except ValueError:
        pass

    raise AttributeError
