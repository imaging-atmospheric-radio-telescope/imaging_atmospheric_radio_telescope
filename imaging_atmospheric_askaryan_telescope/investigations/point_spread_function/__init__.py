from . import plot

from ... import telescope

import spherical_coordinates
import numpy as np
import os
import json_utils
import rename_after_writing as rnw


def init_work_dir(work_dir, telescope_key):
    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    scatter = {}
    with rnw.open(os.path.join(config_dir, "scatter"), "wt") as f:
        f.write(json_utils.dumps(scatter, indent=4))


def make_telescope_like_other_but_with_region_of_interest_camera(
    source_azimuth_rad,
    source_zenith_rad,
    other_telescope,
    region_of_interest_rad,
    num_bins,
):
    roi_rad = region_of_interest_rad
    f = other_telescope["mirror"]["focal_length_m"]
    px_center_rad, py_center_rad = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=source_azimuth_rad,
        zenith_rad=source_zenith_rad,
    )
    cx_center_rad = -px_center_rad
    cy_center_rad = -py_center_rad

    x_bin_edges_m = f * np.linspace(
        cx_center_rad - roi_rad / 2,
        cx_center_rad + roi_rad / 2,
        num_bins + 1,
    )
    y_bin_edges_m = f * np.linspace(
        cy_center_rad - roi_rad / 2,
        cy_center_rad + roi_rad / 2,
        num_bins + 1,
    )

    sensor_roi = telescope.make_sensor_in_region_of_interest(
        x_bin_edges_m=x_bin_edges_m,
        y_bin_edges_m=y_bin_edges_m,
        sensor_distance_m=other_telescope["sensor"]["sensor_distance_m"],
        feed_horn_transmission=1.0,
    )

    return telescope.make_telescope_like_other_but_different_sensor(
        telescope=other_telescope,
        sensor=sensor_roi,
    )


def calculate_total_energy_in_electric_fields(
    E,
    channel_effective_area_m2,
    component_mask=None,
):
    E_magnitude_V_per_m = E.norm_components(component_mask=component_mask)
    P_W = signal.calculate_antenna_power_W(
        effective_area_m2=channel_effective_area_m2,
        electric_field_V_per_m=E_magnitude_V_per_m[:],
    )
    Ene_J = np.sum(P_W) * E.time_slice_duration_s
    return Ene_J
