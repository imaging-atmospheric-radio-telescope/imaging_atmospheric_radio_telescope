import numpy as np
import os
from . import tarstream


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
        os.path.join(path, "electric_fields_V_per_m.antenna.time.dim.float32"), "wb"
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
        o["time_slice_duration_s"] = np.frombuffer(f.read(), dtype="float64")[0]

    with open(os.path.join(path, "num_time_slices.uint64"), "rb") as f:
        o["num_time_slices"] = np.frombuffer(f.read(), dtype="uint64")[0]

    with open(os.path.join(path, "num_antennas.uint64"), "rb") as f:
        o["num_antennas"] = np.frombuffer(f.read(), dtype="uint64")[0]

    with open(os.path.join(path, "global_start_time_s.float64"), "rb") as f:
        o["global_start_time_s"] = np.frombuffer(f.read(), dtype="float64")[0]

    with open(
        os.path.join(path, "electric_fields_V_per_m.antenna.time.dim.float32"), "rb"
    ) as f:
        arr = np.frombuffer(f.read(), dtype="float32")
        o["electric_fields_V_per_m"] = arr.reshape(
            o["num_antennas"], o["num_time_slices"], 3,
        )

    return o


def read_tar(path):
    o = {}
    with tarstream.TarStream(path=path, mode="r") as t:
        filename, filebytes = t.read()
        assert filename == "time_slice_duration_s.float64"
        o["time_slice_duration_s"] = np.frombuffer(filebytes, dtype="float64")[0]

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
            o["num_antennas"], o["num_time_slices"], 3,
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
        electric_fields["electric_fields_V_per_m"][:, :, component_mask], axis=2
    )
