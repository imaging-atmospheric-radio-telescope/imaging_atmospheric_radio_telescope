import numpy as np
import copy
import json_line_logger
import logging


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


class LoggerStartStop:
    def __init__(
        self, start_msg, logger=None, stop_msg="Done.", level=logging.DEBUG
    ):
        self.logger = stdout_logger_if_None(logger)
        self.level = level
        self.start_msg = start_msg
        self.stop_msg = stop_msg

    def __enter__(self):
        self.logger.log(level=self.level, msg=self.start_msg)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.logger.log(level=self.level, msg=self.stop_msg)
        return

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


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


def stdout_logger_if_None(logger):
    return json_line_logger.LoggerStdout() if logger is None else logger
