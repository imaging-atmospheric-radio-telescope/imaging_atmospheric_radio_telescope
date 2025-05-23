import numpy as np


def _decimals(x):
    if x > 1.0:
        return 0
    else:
        return int(np.ceil(np.abs(np.log10(x))))


def _format(decimals):
    return "{:" + f".{decimals:d}f" + "}"


def uncertainty(x, dx):
    decimals = _decimals(dx)
    sx = _format(decimals).format(x)
    sdx = _format(decimals).format(dx)
    return sx + r"\pm" + sdx
