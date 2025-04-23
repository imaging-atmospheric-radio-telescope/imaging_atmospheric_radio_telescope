import numpy as np
import os
from .utils import tarstream
from . import corsika


def init(
    time_slice_duration_s,
    num_time_slices,
    num_antennas,
    global_start_time_s,
):
    assert time_slice_duration_s > 0.0
    assert num_time_slices >= 0
    assert num_antennas >= 0
    out = {}
    out["global_start_time_s"] = global_start_time_s
    out["time_slice_duration_s"] = time_slice_duration_s
    out["num_time_slices"] = num_time_slices
    out["num_antennas"] = num_antennas
    out["electric_fields_V_per_m"] = np.zeros(
        shape=(out["num_antennas"], out["num_time_slices"], 3),
        dtype=np.float32,
    )
    return out


def init_random(seed):
    prng = prng = np.random.Generator(np.random.PCG64(seed))

    E = init(
        time_slice_duration_s=prng.uniform(low=1e-9, high=1e-6),
        num_time_slices=prng.integers(low=100, high=1_000),
        num_antennas=prng.integers(low=10, high=1_000),
        global_start_time_s=prng.uniform(low=-5e-6, high=5e-6),
    )
    E["electric_fields_V_per_m"] = (
        prng.uniform(
            low=-1.0,
            high=1.0,
            size=np.prod(E["electric_fields_V_per_m"].shape),
        )
        .reshape(E["electric_fields_V_per_m"].shape)
        .astype(np.float32)
    )

    return E


def assert_valid(electric_fields):
    E = electric_fields
    assert not np.isnan(E["global_start_time_s"])
    assert E["time_slice_duration_s"] > 0.0
    assert E["num_time_slices"] >= 0
    assert E["num_antennas"] >= 0
    assert E["electric_fields_V_per_m"].shape[0] == E["num_antennas"]
    assert E["electric_fields_V_per_m"].shape[1] == E["num_time_slices"]
    assert E["electric_fields_V_per_m"].shape[2] == 3


def assert_almost_equal(actual, desired, **kwargs):
    assert_valid(actual)
    assert_valid(desired)

    np.testing.assert_almost_equal(
        actual=actual["global_start_time_s"],
        desired=desired["global_start_time_s"],
        **kwargs,
    )
    np.testing.assert_almost_equal(
        actual=actual["time_slice_duration_s"],
        desired=desired["time_slice_duration_s"],
        **kwargs,
    )
    assert actual["num_time_slices"] == desired["num_time_slices"]
    assert actual["num_antennas"] == desired["num_antennas"]
    np.testing.assert_almost_equal(
        actual=actual["electric_fields_V_per_m"],
        desired=desired["electric_fields_V_per_m"],
        **kwargs,
    )


def init_from_coreas_electric_fields(coreas_electric_fields):
    raw = coreas_electric_fields
    time_slice_duration_s = (
        corsika.coreas.coreas_electric_fields.estimate_time_slice_duration_s(
            coreas_electric_fields=raw
        )
    )
    corsika.coreas.coreas_electric_fields.assert_same_time_slice_duration(
        coreas_electric_fields=raw,
        time_slice_duration_s=time_slice_duration_s,
    )
    num_antennas = len(raw)
    num_time_slices = raw[0].shape[0]

    global_start_time_s = np.min(
        [raw[a]["time_s"] for a in range(num_antennas)]
    )

    start_time_offsets_s = np.array(
        [
            raw[a]["time_s"][0] - global_start_time_s
            for a in range(num_antennas)
        ]
    )

    start_slice_offsets_s = np.round(
        start_time_offsets_s / time_slice_duration_s
    ).astype(np.int64)
    assert np.all(start_slice_offsets_s == 0)

    E_V_per_m = np.zeros(
        shape=(num_antennas, num_time_slices, 3),
        dtype=np.float32,
    )

    CGS_TO_SI = (
        corsika.coreas.coreas_electric_fields.CGS_statVolt_per_cm_to_SI_Volt_per_meter
    )
    for a in range(num_antennas):
        E_V_per_m[a, :, 0] = raw[a]["E_north_statVolt_per_cm"] * CGS_TO_SI
        E_V_per_m[a, :, 1] = raw[a]["E_west_statVolt_per_cm"] * CGS_TO_SI
        E_V_per_m[a, :, 2] = raw[a]["E_vertical_statVolt_per_cm"] * CGS_TO_SI

    return {
        "time_slice_duration_s": time_slice_duration_s,
        "num_time_slices": num_time_slices,
        "num_antennas": num_antennas,
        "electric_fields_V_per_m": E_V_per_m,
        "global_start_time_s": global_start_time_s,
    }


def to_coreas_electric_fields(electric_fields):
    ef = electric_fields
    raw = corsika.coreas.coreas_electric_fields.init(
        num_antennas=ef["num_antennas"],
        num_time_slices=ef["num_time_slices"],
    )
    CGS_TO_SI = (
        corsika.coreas.coreas_electric_fields.CGS_statVolt_per_cm_to_SI_Volt_per_meter
    )

    for a in range(ef["num_antennas"]):
        raw[a]["time_s"] = ef["global_start_time_s"] + np.linspace(
            0,
            ef["time_slice_duration_s"] * ef["num_time_slices"],
            ef["num_time_slices"],
        )

        raw[a]["E_north_statVolt_per_cm"] = (
            ef["electric_fields_V_per_m"][a, :, 0] / CGS_TO_SI
        )
        raw[a]["E_west_statVolt_per_cm"] = (
            ef["electric_fields_V_per_m"][a, :, 1] / CGS_TO_SI
        )
        raw[a]["E_vertical_statVolt_per_cm"] = (
            ef["electric_fields_V_per_m"][a, :, 2] / CGS_TO_SI
        )

    return raw


def write(path, electric_fields):
    s = electric_fields
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "time_slice_duration_s.float64"), "wb") as f:
        f.write(np.float64(s["time_slice_duration_s"]).tobytes())

    with open(os.path.join(path, "num_time_slices.uint64"), "wb") as f:
        f.write(np.uint64(s["num_time_slices"]).tobytes())

    with open(os.path.join(path, "num_antennas.uint64"), "wb") as f:
        f.write(np.uint64(s["num_antennas"]).tobytes())

    with open(os.path.join(path, "global_start_time_s.float64"), "wb") as f:
        f.write(np.float64(s["global_start_time_s"]).tobytes())

    assert s["electric_fields_V_per_m"].dtype == np.float32
    with open(
        os.path.join(path, "electric_fields_V_per_m.antenna.time.dim.float32"),
        "wb",
    ) as f:
        f.write(s["electric_fields_V_per_m"].tobytes(order="C"))


def write_tar(path, electric_fields):
    s = electric_fields
    assert s["electric_fields_V_per_m"].dtype == np.float32

    with tarstream.TarStream(path=path, mode="w") as t:
        t.write(
            filename="time_slice_duration_s.float64",
            filebytes=np.float64(s["time_slice_duration_s"]).tobytes(),
        )
        t.write(
            filename="num_time_slices.uint64",
            filebytes=np.uint64(s["num_time_slices"]).tobytes(),
        )
        t.write(
            filename="num_antennas.uint64",
            filebytes=np.uint64(s["num_antennas"]).tobytes(),
        )
        t.write(
            filename="global_start_time_s.float64",
            filebytes=np.float64(s["global_start_time_s"]).tobytes(),
        )
        t.write(
            filename="electric_fields_V_per_m.antenna.time.dim.float32",
            filebytes=s["electric_fields_V_per_m"].tobytes(order="C"),
        )


def read(path):
    o = {}
    with open(os.path.join(path, "time_slice_duration_s.float64"), "rb") as f:
        o["time_slice_duration_s"] = np.frombuffer(f.read(), dtype="float64")[
            0
        ]

    with open(os.path.join(path, "num_time_slices.uint64"), "rb") as f:
        o["num_time_slices"] = np.frombuffer(f.read(), dtype="uint64")[0]

    with open(os.path.join(path, "num_antennas.uint64"), "rb") as f:
        o["num_antennas"] = np.frombuffer(f.read(), dtype="uint64")[0]

    with open(os.path.join(path, "global_start_time_s.float64"), "rb") as f:
        o["global_start_time_s"] = np.frombuffer(f.read(), dtype="float64")[0]

    with open(
        os.path.join(path, "electric_fields_V_per_m.antenna.time.dim.float32"),
        "rb",
    ) as f:
        arr = np.frombuffer(f.read(), dtype="float32")
        o["electric_fields_V_per_m"] = arr.reshape(
            o["num_antennas"],
            o["num_time_slices"],
            3,
        )

    return o


def read_tar(path):
    o = {}
    with tarstream.TarStream(path=path, mode="r") as t:
        filename, filebytes = t.read()
        assert filename == "time_slice_duration_s.float64"
        o["time_slice_duration_s"] = np.frombuffer(filebytes, dtype="float64")[
            0
        ]

        filename, filebytes = t.read()
        assert filename == "num_time_slices.uint64"
        o["num_time_slices"] = np.frombuffer(filebytes, dtype="uint64")[0]

        filename, filebytes = t.read()
        assert filename == "num_antennas.uint64"
        o["num_antennas"] = np.frombuffer(filebytes, dtype="uint64")[0]

        filename, filebytes = t.read()
        assert filename == "global_start_time_s.float64"
        o["global_start_time_s"] = np.frombuffer(filebytes, dtype="float64")[0]

        filename, filebytes = t.read()
        assert filename == "electric_fields_V_per_m.antenna.time.dim.float32"
        tmp = np.frombuffer(filebytes, dtype="float32")
        o["electric_fields_V_per_m"] = tmp.reshape(
            o["num_antennas"],
            o["num_time_slices"],
            3,
        )

    return o


def rotate_electric_field():
    pass


def make_time_bin_edges(electric_fields, global_time=True):
    ef = electric_fields
    exposure_time = ef["num_time_slices"] * ef["time_slice_duration_s"]
    time_bin_edges = np.linspace(0, exposure_time, ef["num_time_slices"] + 1)
    if global_time:
        time_bin_edges = time_bin_edges + ef["global_start_time_s"]
    return time_bin_edges


def make_antenna_bin_edges(electric_fields):
    N = electric_fields["num_antennas"]
    return np.linspace(0.0, N, N + 1) - 0.5


def get_combined_norm_of_components(electric_fields, component_mask):
    assert len(component_mask) == 3
    for comp in component_mask:
        assert comp in [0, 1]
    return np.linalg.norm(
        electric_fields["electric_fields_V_per_m"][:, :, component_mask],
        axis=2,
    )


def estimate_time_of_first_non_zero_amplitudes(electric_fields):
    """
    Estimates the time when the electric fields start do differ from zero.

    Parameters
    ----------
    electric_field : dict

    Returns
    -------
    start_time_s : float
    """
    e = electric_fields
    first_slices = []
    for ant in range(e["num_antennas"]):
        for dim in range(3):
            first_slice = np.min(
                np.nonzero(e["electric_fields_V_per_m"][ant, :, dim])
            )
            first_slices.append(first_slice)

    start_slice = np.median(first_slices)
    start_time_relative = start_slice * e["time_slice_duration_s"]
    start_time_s = start_time_relative + e["global_start_time_s"]
    return start_time_s
