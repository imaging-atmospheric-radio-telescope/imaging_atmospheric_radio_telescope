import numpy as np
import copy
import importlib
from scipy.stats.qmc import Sobol
from scipy.optimize import curve_fit
from . import scientific
from . import sky_and_screen


def package_path():
    return str(
        importlib.resources.files("imaging_atmospheric_askaryan_telescope")
    )


def gauss_pseudo_2d(xy, x0, y0, sigma):
    dx = xy[:, 0] - x0
    dy = xy[:, 1] - y0
    dd = np.hypot(dx, dy)
    return gauss1d(x=dd, x0=0, sigma=sigma)


def gauss1d(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma**2))


class QuasiRandomGenerator:
    """
    In contrast to a pseudo random number generator, this quasi
    generator has a strategy to populate uniform distributions
    as quickly as possible.
    """

    def __init__(self, seed=1):
        self.sobol = Sobol(d=1, rng=seed)

    def uniform(self, low=0, high=1, size=None):
        u = self._uniform_0_1(size=size)
        width = high - low
        return low + u * width

    def _uniform_0_1(self, size=None):
        is_scalar = True if size is None else False
        if size is None:
            size = 1
        out = self.sobol.random(n=size).reshape((size))
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
