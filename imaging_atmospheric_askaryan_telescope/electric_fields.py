import numpy as np
import os


def write(path, electric_fields):
    s = electric_fields
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "time_slice_duration.float64"), "wb") as f:
        f.write(np.float64(s["time_slice_duration"]).tobytes())

    with open(os.path.join(path, "num_time_slices.uint64"), "wb") as f:
        f.write(np.uint64(s["num_time_slices"]).tobytes())

    with open(os.path.join(path, "num_antennas.uint64"), "wb") as f:
        f.write(np.uint64(s["num_antennas"]).tobytes())

    with open(os.path.join(path, "global_start_time.float64"), "wb") as f:
        f.write(np.float64(s["global_start_time"]).tobytes())

    assert s["electric_fields"].dtype == np.float32
    with open(
        os.path.join(path, "electric_fields.antenna.time.dim.float32"), "wb"
    ) as f:
        f.write(s["electric_fields"].tobytes(order="C"))


def read(path):
    o = {}
    with open(os.path.join(path, "time_slice_duration.float64"), "rb") as f:
        o["time_slice_duration"] = np.frombuffer(f.read(), dtype="float64")[0]

    with open(os.path.join(path, "num_time_slices.uint64"), "rb") as f:
        o["num_time_slices"] = np.frombuffer(f.read(), dtype="uint64")[0]

    with open(os.path.join(path, "num_antennas.uint64"), "rb") as f:
        o["num_antennas"] = np.frombuffer(f.read(), dtype="uint64")[0]

    with open(os.path.join(path, "global_start_time.float64"), "rb") as f:
        o["global_start_time"] = np.frombuffer(f.read(), dtype="float64")[0]

    with open(
        os.path.join(path, "electric_fields.antenna.time.dim.float32"), "rb"
    ) as f:
        arr = np.frombuffer(f.read(), dtype="float32")
        o["electric_fields"] = arr.reshape(
            o["num_antennas"], o["num_time_slices"], 3,
        )

    return o


def rotate_electric_field():
    pass
