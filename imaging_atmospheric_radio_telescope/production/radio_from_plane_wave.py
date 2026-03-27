# Copyright 2025 Sebastian A. Mueller
import os
import numpy as np

from .. import time_series
from .. import calibration_source


def make_config():
    config = {}
    config["__type__"] = "plane_wave"
    config["plane_waves"] = {}
    config["plane_waves"][
        "first"
    ] = calibration_source.plane_wave_in_far_field.make_config()
    return config


def simulate_mirror_electric_fields_of_single_plane_wave(
    out_dir,
    plane_wave_config,
    time_slice_duration_s,
    antenna_positions_obslvl_m,
    observation_level_asl_m,
):
    antenna_position_vectors_in_asl_frame_m = antenna_positions_obslvl_m.copy()
    antenna_position_vectors_in_asl_frame_m[:, 2] += observation_level_asl_m

    geometry_setup = calibration_source.plane_wave_in_far_field.make_geometry_setup(
        antenna_position_vectors_in_asl_frame_m=antenna_position_vectors_in_asl_frame_m,
        **plane_wave_config["geometry"],
    )

    power_setup = calibration_source.plane_wave_in_far_field.make_power_setup(
        **plane_wave_config["power"]
    )

    E = calibration_source.plane_wave_in_far_field.plane_wave_in_far_field(
        geometry_setup=geometry_setup,
        power_setup=power_setup,
        sine_wave=plane_wave_config["sine_wave"],
        time_slice_duration_s=time_slice_duration_s,
    )

    os.makedirs(out_dir, exist_ok=True)
    electric_field_path = os.path.join(out_dir, "electric_fields.tar")
    time_series.write(path=electric_field_path, time_series=E)


def simulate_mirror_electric_fields(
    mirror_dir,
    plane_waves,
    time_slice_duration_s,
    antenna_positions_obslvl_m,
    observation_level_asl_m,
):
    os.makedirs(mirror_dir, exist_ok=True)
    plane_waves_dir = os.path.join(mirror_dir, "plane_waves")

    os.makedirs(plane_waves_dir, exist_ok=True)

    t_starts = []
    t_stops = []
    for key in plane_waves:
        plane_wave_config = plane_waves[key]
        plane_wave_dir = os.path.join(plane_waves_dir, key)
        simulate_mirror_electric_fields_of_single_plane_wave(
            out_dir=plane_wave_dir,
            plane_wave_config=plane_wave_config,
            time_slice_duration_s=time_slice_duration_s,
            antenna_positions_obslvl_m=antenna_positions_obslvl_m,
            observation_level_asl_m=observation_level_asl_m,
        )

        E_key = time_series.read(
            os.path.join(plane_wave_dir, "electric_fields.tar")
        )

        t_starts.append(E_key.global_start_time_s)
        t_stops.append(E_key.global_start_time_s + E_key.exposure_duration_s)
        assert E_key.si_unit == "V_per_m"
        assert E_key.dtype == "float32"

    t_start_s = np.min(t_starts)
    t_stop_s = np.max(t_stops)
    duration_s = t_stop_s - t_start_s
    num_time_slices = int(np.ceil(duration_s / time_slice_duration_s))

    E_mirror = time_series.zeros(
        time_slice_duration_s=time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=len(antenna_positions_obslvl_m),
        num_components=3,
        global_start_time_s=t_start_s,
        si_unit="V_per_m",
        dtype="f4",
    )

    for key in plane_waves:
        plane_wave_dir = os.path.join(plane_waves_dir, key)
        E_key = time_series.read(
            os.path.join(plane_wave_dir, "electric_fields.tar")
        )
        E_mirror = E_mirror.add(E_key)

    electric_field_path = os.path.join(mirror_dir, "electric_fields.tar")
    time_series.write(path=electric_field_path, time_series=E_mirror)
