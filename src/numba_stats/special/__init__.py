"""Universal functions.

Port of special functions from cephes C library (https://github.com/jeremybarnes/cephes)
or Scipy C implementations (see scipy.special).
"""

__all__ = [
    "erfinv",
    "ndtri",
    "polevl",
]

from ._polevl import polevl
from ._ndtri import ndtri
from ._erfinv import erfinv
