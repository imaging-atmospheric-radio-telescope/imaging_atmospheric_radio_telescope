# Copyright 2025 Sebastian A. Mueller
import os

from .. import time_series
from .. import calibration_source


def make_config():
    return calibration_source.plane_wave_in_far_field.make_config()


def simulate_mirror_electric_fields(
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

    mirror_dir = os.path.join(out_dir, "mirror")
    os.makedirs(mirror_dir, exist_ok=True)
    antenna_path = os.path.join(mirror_dir, "electric_fields.tar")
    time_series.write(path=antenna_path, time_series=E)
