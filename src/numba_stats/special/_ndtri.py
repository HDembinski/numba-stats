"""Port of cephes ``ndtri.c``.

See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtri.c
"""

import numba as nb
import numpy as np
from numpy import pi
from ._polevl import polevl
from .._util import _Floats


@nb.njit([T(T) for T in _Floats], cache=True, inline="never", error_model="numpy")
def ndtri(y):
    """Inverse of Normal distribution function."""
    T = type(y)
    root_2_pi = np.sqrt(T(2 * pi))
    exp_neg2 = np.exp(T(-2))

    # approximation for 0 <= abs(y - 0.5) <= 3/8
    p0 = np.array(
        [
            -1.23916583867381258016e0,
            1.39312609387279679503e1,
            -5.66762857469070293439e1,
            9.80010754185999661536e1,
            -5.99633501014107895267e1,
        ]
    )

    q0 = np.array(
        [
            -1.18331621121330003142e0,
            1.59056225126211695515e1,
            -8.20372256168333339912e1,
            2.00260212380060660359e2,
            -2.25462687854119370527e2,
            8.63602421390890590575e1,
            4.67627912898881538453e0,
            1.95448858338141759834e0,
            1.0,
        ]
    )

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
    # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
    p1 = np.array(
        [
            -8.57456785154685413611e-4,
            -3.50424626827848203418e-2,
            -1.40256079171354495875e-1,
            2.18663306850790267539e0,
            1.46849561928858024014e1,
            4.40805073893200834700e1,
            5.71628192246421288162e1,
            3.15251094599893866154e1,
            4.05544892305962419923e0,
        ]
    )

    q1 = np.array(
        [
            -9.33259480895457427372e-4,
            -3.80806407691578277194e-2,
            -1.42182922854787788574e-1,
            2.50464946208309415979e0,
            1.50425385692907503408e1,
            4.13172038254672030440e1,
            4.53907635128879210584e1,
            1.57799883256466749731e1,
            1.0,
        ]
    )

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
    # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    p2 = np.array(
        [
            6.23974539184983293730e-9,
            2.65806974686737550832e-6,
            3.01581553508235416007e-4,
            1.23716634817820021358e-2,
            2.01485389549179081538e-1,
            1.33303460815807542389e0,
            3.93881025292474443415e0,
            6.91522889068984211695e0,
            3.23774891776946035970e0,
        ]
    )

    q2 = np.array(
        [
            6.79019408009981274425e-9,
            2.89247864745380683936e-6,
            3.28014464682127739104e-4,
            1.34204006088543189037e-2,
            2.16236993594496635890e-1,
            1.37702099489081330271e0,
            3.67983563856160859403e0,
            6.02427039364742014255e0,
            1.0,
        ]
    )

    sign_flag = 1

    if y > (T(1) - exp_neg2):
        y = T(1) - y
        sign_flag = 0

    # Shortcut case where we don't need high precision
    # between -0.135 and 0.135
    if y > exp_neg2:
        y -= T(0.5)
        y2 = y**2
        x = y + y * (y2 * polevl(y2, p0) / polevl(y2, q0))
        x = x * root_2_pi
        return x

    x = np.sqrt(T(-2) * np.log(y))
    z = np.reciprocal(x)
    x0 = x - np.log(x) * z

    if x < T(8.0):  # y > exp(-32) = 1.2664165549e-14
        x1 = z * polevl(z, p1) / polevl(z, q1)
    else:
        x1 = z * polevl(z, p2) / polevl(z, q2)

    x = x0 - x1
    if sign_flag != 0:
        x = -x

    return x
