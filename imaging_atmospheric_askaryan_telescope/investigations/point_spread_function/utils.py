from ... import utils as iaat_utils
from ... import signal
from ... import telescope
from ... import lownoiseblock
from ... import timing_and_sampling

import numpy as np
import json_utils
import os
import copy
import spherical_coordinates


def serial_pool_if_None(pool):
    return iaat_utils.SerialPool() if pool is None else pool


def read_config(work_dir):
    config = json_utils.tree.read(os.path.join(work_dir, "config"))
    config = iaat_utils.strip_dict(config, "comment")
    return config


def substract_one_when_even(x):
    if np.mod(x, 2) > 0:
        return x - 1
    else:
        return x


def make_field_of_view_region_edges(sensor, focal_length_m):
    regions = {}

    _inner_radius_m = (
        sensor["camera"]["outer_radius_m"]
        - sensor["camera"]["feed_horn_inner_radius_m"]
    )
    regions["field_of_view_fully_inside_half_angle_rad"] = np.arctan(
        _inner_radius_m / focal_length_m
    )

    _outer_radius_m = (
        sensor["camera"]["outer_radius_m"]
        + sensor["camera"]["feed_horn_inner_radius_m"]
    )
    regions["field_of_view_fully_outside_half_angle_rad"] = np.arctan(
        _outer_radius_m / focal_length_m
    )

    regions["central_feed_horn_half_angle_rad"] = np.arctan(
        sensor["camera"]["feed_horn_inner_radius_m"] / focal_length_m
    )
    return regions


def make_telescope_timing_and_site(
    config, telescope_key, sensor_distance_m=None
):
    telescope_config = config["telescopes"][telescope_key]

    if sensor_distance_m is not None:
        telescope_config["sensor"]["sensor_distance_m"] = copy.copy(
            sensor_distance_m
        )

    _lnb = lownoiseblock.init(key=telescope_config["lnb_key"])
    _mirror = telescope.make_mirror(**telescope_config["mirror"])
    _sensor = telescope.make_sensor(**telescope_config["sensor"])

    tscope = telescope.make_telescope(
        sensor=_sensor,
        mirror=_mirror,
        lnb=_lnb,
        speed_of_light_m_per_s=signal.SPEED_OF_LIGHT_M_PER_S,
    )
    tscope["key"] = copy.copy(telescope_key)
    timing = timing_and_sampling.make_timing_from_lnb(
        lnb=tscope["lnb"],
        **config["timing_and_sampling"],
    )
    return tscope, timing, config["site"]


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
