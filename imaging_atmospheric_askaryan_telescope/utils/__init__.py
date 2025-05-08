import numpy as np
import copy


class PrintStartStop:
    def __init__(self, start_msg, stop_msg="Done."):
        self.start_msg = start_msg
        self.stop_msg = stop_msg

    def __enter__(self):
        print(
            self.start_msg,
            " ... ",
            end="",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(self.stop_msg)
        return

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


class PrintProgress:
    def __init__(self, num_steps, s="."):
        assert num_steps > 0
        self.s = s
        self.num_steps = num_steps
        self.step = 0
        self.percent = 0

    def __enter__(self):
        return self

    def bump(self):
        self.step += 1
        percent = int(np.round(100 * self.step / self.num_steps))

        if percent > 100:
            print("?", end="", flush=True)
            return

        if percent > self.percent:
            print(self.s, end="", flush=True)
            self.percent = copy.copy(percent)

    def __exit__(self, exc_type, exc_value, exc_traceback):
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
