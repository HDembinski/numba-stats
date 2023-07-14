"""
Cruijff distribution.

See For example: https://arxiv.org/abs/1005.4087
"""

from ._util import _jit, _generate_wrappers, _prange
import numpy as np

_doc_par = """
x : ArrayLike
    Random variate.
beta_left: float
    Left tail acceleration parameter.
beta_right: float
    Right tail acceleration parameter.
loc : float
    Location of the maximum of the distribution.
scale_left : float
    Left width parameter.
scale_right: float
    Right width parameter.
"""


@_jit(5)
def _density(x, beta_left, beta_right, loc, scale_left, scale_right):
    r = np.empty_like(x)
    for i in _prange(len(x)):
        if x[i] < loc:
            scale = scale_left
            beta = beta_left
        else:
            scale = scale_right
            beta = beta_right
        z = (x[i] - loc) / scale
        r[i] = -0.5 * z**2 / (1 + beta * z**2)
    return np.exp(r)


_generate_wrappers(globals())
