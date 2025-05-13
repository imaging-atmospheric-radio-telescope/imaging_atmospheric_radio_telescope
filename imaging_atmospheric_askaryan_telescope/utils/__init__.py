import numpy as np
import copy
from scipy.stats.qmc import Sobol


class QuasiRandomGenerator:
    def __init__(self, low, high, random_seed=1):
        self.sobol = Sobol(d=1, rng=random_seed)
        self.low = low
        self.high = high
        self.range = self.high - self.low

    def uniform(self, size=None):
        is_scalar = True if size is None else False
        if size is None:
            size = 1
        r = self.sobol.random(n=size).reshape((size))
        out = self.low + (r * self.range)
        if is_scalar:
            out = out[0]
        return out


def make_parabola_surface_height_m(
    distance_to_optical_axis_m,
    focal_length_m,
):
    """
    Parameters
    ----------
    distance_to_optical_axis_m : float
        The distance to the parabola's optical axis for where the height of
        the parabola is estimated.
    focal_length_m : float
        The parabola's focal-length.
    """
    z = 1 / (4.0 * focal_length_m) * distance_to_optical_axis_m**2
    return z


class SerialPool:
    def __init__(self):
        pass

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def starmap(self, func, iterable):
        return [func(*item) for item in iterable]

    def __repr__(self):
        out = "{:s}()".format(self.__class__.__name__)
        return out


def strip_dict(obj, strip):
    out = {}
    for key in obj:
        if key != strip:
            item = obj[key]
            if isinstance(item, dict):
                out[key] = strip_dict(obj=item, strip=strip)
            else:
                out[key] = item
    return out


def area_of_hexagon(inner_radius):
    return 2.0 * np.sqrt(3.0) * inner_radius**2.0


def inner_radius_of_hexagon(area):
    return np.sqrt(area / (2.0 * np.sqrt(3.0)))


def _irwin_hall(prng, size=1, order=12):
    arr = prng.uniform(0, 1, size=order * size).reshape((size, order))
    return np.sum(arr, axis=1) - order / 2


def normal_approximation(
    prng,
    mean,
    std,
    size,
    irwin_hall_order=12,
):
    return (
        _irwin_hall(prng=prng, size=size, order=irwin_hall_order) * std + mean
    )


def argmaxNd(a):
    return np.unravel_index(np.argmax(a), a.shape)


def hexagon_inner_radius_given_outer_radius(outer_radius):
    return outer_radius / (2.0 / np.sqrt(3.0))


def hexagon_outer_radius_given_inner_radius(inner_radius):
    return inner_radius * (2.0 / np.sqrt(3.0))
