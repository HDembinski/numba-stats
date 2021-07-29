import numba as nb

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, a, w):
    if a <= x <= a + w:
        return 1 / w
    return 0


@nb.vectorize(_signatures, cache=True)
def cdf(x, a, w):
    if a <= x:
        if x <= a + w:
            return (x - a) / w
        return 1
    return 0


@nb.vectorize(_signatures, cache=True)
def ppf(p, a, w):
    return w * p + a
