from ._util import _vectorize


@_vectorize(3)
def pdf(x, a, w):
    """
    Return probability density of uniform distribution.
    """
    if a <= x <= a + w:
        return 1 / w
    return 0


@_vectorize(3)
def cdf(x, a, w):
    """
    Return cumulative probability of uniform distribution.
    """
    if a <= x:
        if x <= a + w:
            return (x - a) / w
        return 1
    return 0


@_vectorize(3)
def ppf(p, a, w):
    """
    Return quantile of uniform distribution for given probability.
    """
    return w * p + a
