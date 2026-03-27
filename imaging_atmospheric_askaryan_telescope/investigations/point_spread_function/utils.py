from ... import utils as iaat_utils
from ... import signal
from ... import telescope
from ... import calibration
from ... import lownoiseblock
from ... import timing_and_sampling

import numpy as np
import json_utils
import os
import copy
import spherical_coordinates
import binning_utils
import pandas


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


def area_of_sphere(radius):
    return 4.0 * np.pi * radius**2.0


def set_power_with_areal_density(plane_wave_config, power_density_W_per_m2):
    plane_wave_config["power"] = {}
    r_100km = 100e3
    A_sphere_100km = area_of_sphere(radius=r_100km)
    P_isotrop_100km_W = power_density_W_per_m2 * A_sphere_100km
    plane_wave_config["power"][
        "power_of_isotrop_and_point_like_emitter_W"
    ] = P_isotrop_100km_W
    plane_wave_config["power"][
        "distance_to_isotrop_and_point_like_emitter_m"
    ] = r_100km
    return plane_wave_config


def make_field_of_view_region_edges(sensor, focal_length_m):
    regions = {}
    feed_horn_diameter_m = 2.0 * sensor["camera"]["feed_horn_inner_radius_m"]

    _inner_radius_m = sensor["camera"]["outer_radius_m"] - feed_horn_diameter_m
    regions["field_of_view_fully_inside_half_angle_rad"] = np.arctan(
        _inner_radius_m / focal_length_m
    )

    _outer_radius_m = sensor["camera"]["outer_radius_m"] + feed_horn_diameter_m
    regions["field_of_view_fully_outside_half_angle_rad"] = np.arctan(
        _outer_radius_m / focal_length_m
    )

    regions["central_feed_horn_half_angle_rad"] = np.arctan(
        sensor["camera"]["feed_horn_inner_radius_m"] / focal_length_m
    )

    regions["field_of_view_half_angle_rad"] = np.arctan(
        sensor["camera"]["outer_radius_m"] / focal_length_m
    )

    return regions


def make_telescope_timing_and_site(
    work_dir, config, telescope_key, sensor_distance_m=None
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
    tscope = calibration.add_calibration_to_telescope(
        telescope=tscope,
        path=os.path.join(work_dir, "calibration", telescope_key),
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

    tele = telescope.make_telescope_like_other_but_different_sensor(
        telescope=other_telescope,
        sensor=sensor_roi,
    )

    tele["calibration"] = {}
    tele["calibration"][
        "point_spread_function_quantile_contained_in_feed_horn"
    ] = {}
    tele["calibration"][
        "point_spread_function_quantile_contained_in_feed_horn"
    ]["watershed"] = 1.0

    return tele


def make_feed_horns_signal_mask(feed_horn_positions_m, x_m, y_m, r_m):
    mask = np.zeros(feed_horn_positions_m.shape[0], dtype=bool)
    for i in range(feed_horn_positions_m.shape[0]):
        fx, fy, _ = feed_horn_positions_m[i]
        d = np.hypot((fx - x_m), (fy - y_m))
        if d <= r_m:
            mask[i] = True
    return mask


def histogram_p50_s68(x, y, edges):
    num = len(edges) - 1
    p50 = np.zeros(num)
    s68 = np.zeros(num)
    cnt = np.zeros(num)

    for i in range(num):
        start = edges[i]
        stop = edges[i + 1]
        mask = np.logical_and(x >= start, x < stop)
        cnt[i] = np.sum(mask)
        if cnt[i] > 0:
            p50[i] = np.percentile(y[mask], 50)
            s68[i] = percentile_spread(y[mask], 68)
        else:
            p50[i] = float("nan")
            s68[i] = float("nan")
    return {"p50": p50, "s68": s68, "cnt": cnt}


def percentile_spread(x, p):
    p_half = p / 2
    x_start = np.percentile(x, 50 - p_half)
    x_stop = np.percentile(x, 50 + p_half)
    return x_stop - x_start


def fit_poly1d(x, y):
    """
    Fiting:

    y = a*x + b

    Returns
    -------
    [a, b], [a_std, b_std]
    """
    if len(x) > 2:
        ab, ab_cov = np.polyfit(x=x, y=y, deg=1, cov=True)
        ab_std = np.sqrt(np.diag(ab_cov))
    else:
        ab = np.polyfit(x=x, y=y, deg=1, cov=False)
        ab_std = np.array([float("nan"), float("nan")])
    return ab, ab_std


def guess_off_axis_binning(num_samples, half_angle):
    off_num_bins = int(np.sqrt(0.5 * num_samples))
    off_num_bins = np.max([3, off_num_bins])
    oa_bin = binning_utils.Binning(
        bin_edges=np.linspace(
            0.0,
            half_angle**2,
            off_num_bins + 1,
        )
        ** 0.5
    )
    return oa_bin


def read_jsonl_reports_into_recarray(path):
    reports = json_utils.lines.read(path)
    df = pandas.DataFrame.from_records(reports)
    return df.to_records(index=False)
