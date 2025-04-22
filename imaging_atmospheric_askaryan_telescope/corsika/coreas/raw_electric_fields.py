import numpy as np
import os
import glob
import io
import rename_after_writing as rnw


DTYPES = [
    ("time_s", float),
    ("E_north_statVolt_per_cm", float),
    ("E_west_statVolt_per_cm", float),
    ("E_vertical_statVolt_per_cm", float),
]


def _init_raw_electric_field(num_time_slices):
    return np.recarray(
        shape=num_time_slices,
        dtype=DTYPES,
    )


def init(num_antennas, num_time_slices):
    return [
        _init_raw_electric_field(num_time_slices) for i in range(num_antennas)
    ]


def init_random(seed):
    prng = prng = np.random.Generator(np.random.PCG64(seed))

    num_time_slices = prng.integers(low=100, high=1_000)
    num_antennas = prng.integers(low=10, high=100)
    global_start_time_s = prng.uniform(low=-5e-6, high=5e-6)
    time_slice_duration_s = prng.uniform(low=1e-9, high=1e-6)

    raw = init(num_antennas=num_antennas, num_time_slices=num_time_slices)

    for a in range(num_antennas):
        raw[a]["time_s"] = global_start_time_s + np.linspace(
            0,
            time_slice_duration_s * num_time_slices,
            num_time_slices,
        )
        raw[a]["E_north_statVolt_per_cm"] = prng.uniform(
            low=-1, high=+1, size=num_time_slices
        )
        raw[a]["E_west_statVolt_per_cm"] = prng.uniform(
            low=-1, high=+1, size=num_time_slices
        )
        raw[a]["E_vertical_statVolt_per_cm"] = prng.uniform(
            low=-1, high=+1, size=num_time_slices
        )
    return raw


def assert_almost_eqaul(actual, desired, **kwargs):
    assert len(actual) == len(desired)

    for a in range(len(actual)):
        for dtype in DTYPES:
            key = dtype[0]
            np.testing.assert_array_almost_equal(
                actual=actual[a][key], desired=desired[a][key], **kwargs
            )


def estimate_time_slice_duration_s(raw_electric_fields):
    first_antenna = 0
    return np.median(np.gradient(raw_electric_fields[first_antenna]["time_s"]))


def assert_same_time_slice_duration(
    raw_electric_fields, time_slice_duration_s
):
    num_antennas = len(raw_electric_fields)

    for a in range(num_antennas):
        time_slice_duration_this_antenna = np.gradient(
            raw_electric_fields[a]["time_s"]
        )
        valid = (
            np.abs(time_slice_duration_this_antenna - time_slice_duration_s)
            < time_slice_duration_s / 10
        )
        assert np.all(valid)
    return time_slice_duration_s


def list_time_series_paths_in_numerical_order(path):
    all_time_series_paths = glob.glob(os.path.join(path, "raw_*.dat"))
    antenna_indices = []
    for time_series_path in all_time_series_paths:
        basename = os.path.basename(time_series_path)
        antenna_index = int(basename[4:10])
        antenna_indices.append(antenna_index)
    antenna_indices = np.array(antenna_indices)
    order = np.argsort(antenna_indices)
    all_time_series_paths = [all_time_series_paths[i] for i in order]
    return all_time_series_paths


# https://www.unitconverters.net/electric-field-strength/statvolt-centimeter-to-volt-meter.htm
CGS_statVolt_per_cm_to_SI_Volt_per_meter = 2.99792458e4


def read(path):
    """
    Reads the output electric fields written by CORSIKA CoREAS.

    Parameters
    ----------
    path : str
        Output directory where CoREAS writes its 'raw_*.dat' fiels to.

    Returns
    -------
    raw_electric_fields : list of numpy.recarrays
        In CGC units (StatVolt / cm).
    """
    all_time_series_paths = list_time_series_paths_in_numerical_order(path)
    raw_electric_fields = []
    for time_series_path in all_time_series_paths:
        with open(time_series_path, "rt") as f:
            raw_electric_field = loads(text=f.read())
        raw_electric_fields.append(raw_electric_field)
    return raw_electric_fields


def dumps(raw_electric_field):
    """
    Dumps a single raw_electric_field into a string with the format used
    by CORSIKA CoREAS.

    Parameters
    ----------
    raw_electric_field : numpy.recarray
        time, north, west, vertical

    Returns
    -------
    text : str
    """
    num_time_slices = raw_electric_field.shape[0]
    s = io.StringIO()
    for i in range(num_time_slices):
        t, n, w, v = raw_electric_field[i]
        s.write(f"{t:f}\t{n:f}\t{w:f}\t{v:f}\n")
    s.seek(0)
    return s.read()


def loads(text):
    """
    Parameters
    ----------
    text : str

    Returns
    -------
    raw_electric_field : numpy.recarray
        time, north, west, vertical
    """
    tt = []
    nn = []
    ww = []
    vv = []
    for line in str.splitlines(text):
        tokens = str.split(line, "\t")
        t, n, w, v = [float(token) for token in tokens]
        tt.append(t)
        nn.append(n)
        ww.append(w)
        vv.append(v)

    raw_electric_field = _init_raw_electric_field(num_time_slices=len(tt))
    raw_electric_field["time_s"] = tt
    raw_electric_field["E_north_statVolt_per_cm"] = nn
    raw_electric_field["E_west_statVolt_per_cm"] = ww
    raw_electric_field["E_vertical_statVolt_per_cm"] = vv
    return raw_electric_field


def write(path, raw_electric_fields):
    """
    Writes the 'raw_electric_fields' into the directory 'path', the same way as
    CoREAS does it.

    Parameters
    ----------
    path : str
        Output directory.
    raw_electric_fields : list of numpy.recarrays
        In CGC units (StatVolt / cm).
    """
    os.makedirs(path, exist_ok=True)
    num_antennas = len(raw_electric_fields)
    for a in range(num_antennas):
        time_series_path = os.path.join(path, f"raw_{a:06d}.dat")
        raw_electric_field = raw_electric_fields[a]
        with rnw.open(time_series_path, "wt") as f:
            text = dumps(raw_electric_field=raw_electric_field)
            f.write(text)
