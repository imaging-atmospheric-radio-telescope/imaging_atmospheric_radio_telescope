import numpy as np
import os
import copy
import builtins
from .utils import tarstream


def zeros(
    time_slice_duration_s,
    num_time_slices,
    num_channels,
    num_components,
    global_start_time_s,
    si_unit=None,
    dtype="f4",
):
    return TimeSeries(
        time_slice_duration_s=time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=num_channels,
        num_components=num_components,
        global_start_time_s=global_start_time_s,
        si_unit=si_unit,
        dtype=dtype,
    )


def zeros_like(other, num_components=None):

    if num_components is None:
        num_components = other.num_components

    return zeros(
        time_slice_duration_s=other.time_slice_duration_s,
        num_time_slices=other.num_time_slices,
        num_channels=other.num_channels,
        num_components=other.num_components,
        global_start_time_s=other.global_start_time_s,
        si_unit=other.si_unit,
        dtype=other.dtype,
    )


def copy(other):
    out = zeros_like(other)
    out._x = np.copy(other._x)
    return out


class TimeSeries:
    def __init__(
        self,
        time_slice_duration_s,
        num_time_slices,
        num_channels,
        num_components,
        global_start_time_s,
        si_unit=None,
        dtype="f4",
    ):

        assert time_slice_duration_s > 0.0
        assert num_time_slices >= 0
        assert num_channels >= 0
        assert num_components >= 0

        self.global_start_time_s = float(global_start_time_s)
        self.time_slice_duration_s = float(time_slice_duration_s)

        self.num_time_slices = int(num_time_slices)
        self.num_channels = int(num_channels)
        self.num_components = int(num_components)

        self.si_unit = si_unit

        self._x = np.zeros(
            shape=(
                self.num_channels,
                self.num_time_slices,
                self.num_components,
            ),
            dtype=dtype,
        )

    def __getitem__(self, subscript):
        return self._x[subscript]

    def __setitem__(self, subscript):
        self._x[subscript]

    @property
    def dtype(self):
        return self._x.dtype

    def exposure_duration_s(self):
        return self.time_slice_duration_s * self.num_time_slices

    def make_channel_bin_edges(self):
        N = self.num_antennas
        return np.linspace(0.0, N, N + 1) - 0.5

    def make_time_bin_centers(self, global_time=True):
        tt = np.linspace(
            0,
            (self.num_time_slices - 1) * self.time_slice_duration_s,
            self.num_time_slices,
        )
        if global_time:
            tt += self.global_start_time_s
        return tt

    def make_time_bin_edges(self, global_time=True):
        time_bin_edges_s = np.linspace(
            0,
            self.exposure_duration_s,
            self.num_time_slices + 1,
        )
        if global_time:
            time_bin_edges_s = time_bin_edges_s + self.global_start_time_s
        return time_bin_edges_s

    def norm_components(self):
        """
        Returns the norm along the channel components.
        """
        out = zeros_like(self, num_components=1)
        out._x = np.linalg.norm(self._x[:, :, :], axis=2)
        return out

    def sum_components(self):
        out = zeros_like(self, num_components=1)
        out._x = np.sum(self._x[:, :, :], axis=2)
        return out

    def add(self, other):
        """
        Add up the electric fields in other and self according to their
        global start times. Output will only be when self and other overlap
        in time.

                    |-------------| other
                    .     |--------------------------| self
                    .     .
                    T1    T2

        Returns
        -------
        modified copy of self
        """
        assert other.num_channels == self.num_channels

        if other.time_slice_duration_s == self.time_slice_duration_s:
            dT_s = other.time_slice_duration_s
        else:
            dT1_s = other.time_slice_duration_s
            dT2_s = self.time_slice_duration_s
            epsilon = 1e-6
            assert (1.0 - epsilon) < dT1_s / dT2_s < (1.0 + epsilon)
            dT_s = np.mean([dT1_s, dT2_s])

        delta_T_s = other.global_start_time_s - self.global_start_time_s
        at_time_slice_in_second = int(np.round(delta_T_s / dT_s))

        out = copy(self)
        for channel in range(other.num_channels):
            signal.add_first_to_second_at(
                first=other._x[channel],
                second=out.x[channel],
                at=at_time_slice_in_second,
            )

        return out

    def write(self, path):
        return write_tar(path=path, time_series=self)


def assert_valid(time_series):
    E = time_series
    assert not np.isnan(E.global_start_time_s)
    assert E.time_slice_duration_s > 0.0
    assert E.num_time_slices >= 0
    assert E.num_channels >= 0
    assert E.num_components >= 0
    assert E._x.shape[0] == E.num_channels
    assert E._x.shape[1] == E.num_time_slices
    assert E._x.shape[2] == E.num_components


def assert_almost_equal(actual, desired, **kwargs):
    assert_valid(actual)
    assert_valid(desired)

    np.testing.assert_almost_equal(
        actual=actual.global_start_time_s,
        desired=desired.global_start_time_s,
        **kwargs,
    )
    np.testing.assert_almost_equal(
        actual=actual.time_slice_duration_s,
        desired=desired.time_slice_duration_s,
        **kwargs,
    )
    assert actual.num_time_slices == desired.num_time_slices
    assert actual.num_channels == desired.num_channels
    assert actual.num_components == desired.num_components

    assert actual.si_unit == desired.si_unit
    np.testing.assert_almost_equal(
        actual=actual._x,
        desired=desired._x,
        **kwargs,
    )


def write(path, time_series):
    s = time_series

    with tarstream.TarStream(path=path, mode="w") as t:
        t.write(
            filename="time_slice_duration_s.float64",
            filebytes=np.float64(s.time_slice_duration_s).tobytes(),
        )
        t.write(
            filename="num_time_slices.uint64",
            filebytes=np.uint64(s.num_time_slices).tobytes(),
        )
        t.write(
            filename="num_channels.uint64",
            filebytes=np.uint64(s.num_channels).tobytes(),
        )
        t.write(
            filename="num_components.uint64",
            filebytes=np.uint64(s.num_components).tobytes(),
        )
        t.write(
            filename="global_start_time_s.float64",
            filebytes=np.float64(s.global_start_time_s).tobytes(),
        )
        t.write(
            filename=f"si_unit.txt",
            filebytes=s.si_unit.encode(),
        )
        t.write(
            filename=f"x.channel.time.component.{s._x.dtype.name:s}",
            filebytes=s._x.tobytes(order="C"),
        )


def read(path):
    with tarstream.TarStream(path=path, mode="r") as t:
        filename, filebytes = t.read()
        assert filename == "time_slice_duration_s.float64"
        time_slice_duration_s = np.frombuffer(filebytes, dtype="float64")[0]

        filename, filebytes = t.read()
        assert filename == "num_time_slices.uint64"
        num_time_slices = int(np.frombuffer(filebytes, dtype="uint64")[0])

        filename, filebytes = t.read()
        assert filename == "num_channels.uint64"
        num_channels = int(np.frombuffer(filebytes, dtype="uint64")[0])

        filename, filebytes = t.read()
        assert filename == "num_components.uint64"
        num_components = int(np.frombuffer(filebytes, dtype="uint64")[0])

        filename, filebytes = t.read()
        assert filename == "global_start_time_s.float64"
        global_start_time_s = np.frombuffer(filebytes, dtype="float64")[0]

        filename, filebytes = t.read()
        assert filename == "si_unit.txt"
        si_unit = filebytes.decode()

        filename, filebytes = t.read()
        assert filename.startswith("x.channel.time.component.")
        _dtype_ext = os.path.splitext(filename)[-1]
        assert _dtype_ext.startswith(".")
        dtype_name = _dtype_ext[1:]

        o = zeros(
            time_slice_duration_s=time_slice_duration_s,
            num_time_slices=num_time_slices,
            num_channels=num_channels,
            num_components=num_components,
            global_start_time_s=global_start_time_s,
            dtype=dtype_name,
            si_unit=si_unit,
        )

        tmp = np.frombuffer(filebytes, dtype=dtype_name)
        o._x = tmp.reshape(
            o.num_channels,
            o.num_time_slices,
            o.num_components,
        )

    return o


def print(time_series, num_samples_to_be_integrated=20, channels=None):
    """
    Print the amplitudes with a bar graph.

    Parameters
    ----------
    time_series : time_series.TimeSeries

    num_samples_to_be_integrated : int
        To not print all time slices, we integrate over this many.
    channels : array like (default: None)
        List of channel indices to be printed. If None, all channels will
        be printed.
    """
    N = num_samples_to_be_integrated
    E = time_series
    if channels is None:
        channels = np.arange(0, E.num_channels)

    MM = []
    TT = []
    for b in range(E.num_time_slices // N):
        s_start = b * N
        TT.append(s_start * E.time_slice_duration_s)

    for a in channels:
        e = np.linalg.norm(E._x[a], axis=1)
        emax = np.max(e)
        mm = []
        for b in range(E.num_time_slices // N):
            s_start = b * N
            s_stop = s_start + N
            m = np.mean(e[s_start:s_stop]) / emax
            mm.append(m)
        MM.append(mm)
    MM = np.array(MM)

    head = "time/ns "
    for a in channels:
        head += f"{a: 10d} "
    builtins.print(head)

    for t in range(len(TT)):
        line = f"{TT[t]*1e9: 7.1f} "
        for a in channels:
            nn = int(MM[a][t] * 10)
            for i in range(10):
                if i < nn:
                    line += "|"
                else:
                    line += "."
            line += " "
        builtins.print(line)


def random(seed):
    prng = prng = np.random.Generator(np.random.PCG64(seed))

    E = zeros(
        time_slice_duration_s=prng.uniform(low=1e-9, high=1e-6),
        num_time_slices=prng.integers(low=100, high=1_000),
        num_channels=prng.integers(low=10, high=1_000),
        num_components=prng.integers(low=1, high=3),
        global_start_time_s=prng.uniform(low=-5e-6, high=5e-6),
        si_unit=_draw_random_printable_string(prng=prng, size=6),
    )
    E._x = (
        prng.uniform(
            low=-1.0,
            high=1.0,
            size=np.prod(E._x.shape),
        )
        .reshape(E._x.shape)
        .astype(E.dtype)
    )
    return E


def _draw_random_printable_string(prng, size):
    return prng.integers(65, 90, size).astype(np.uint8).tobytes().decode()
