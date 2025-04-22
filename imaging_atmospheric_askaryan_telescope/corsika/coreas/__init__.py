import numpy as np
import glob
import os
from . import calibration_source
from . import antenna_list

DEFAULT_TIME_BOUNDARIES = {
    "automatic_time_boundaries_s": 4e-07,
    "time_lower_boundary_s": -1,
    "time_upper_boundary_s": 1,
}


def make_steering_card(
    core_position_on_observation_level_north_m,
    core_position_on_observation_level_west_m,
    core_position_on_observation_level_asl_m,
    time_slice_duration_s,
    time_boundaries=DEFAULT_TIME_BOUNDARIES,
):
    core_north_cm = core_position_on_observation_level_north_m * 1e2
    core_west_cm = core_position_on_observation_level_west_m * 1e2
    cors_asl_cm = core_position_on_observation_level_asl_m * 1e2

    sc = "# CoREAS V1.1 by Tim Huege <tim.huege@kit.edu> with contributions "
    sc += "by Marianne Ludwig and Clancy James - parameter file\n"
    sc += "\n"
    sc += "# parameters setting up the spatial observer configuration:\n"
    sc += "\n"
    sc += "CoreCoordinateNorth = {0:.6E} ; in cm\n".format(core_north_cm)
    sc += "CoreCoordinateWest = {0:.6E} ; in cm\n".format(core_west_cm)
    sc += "CoreCoordinateVertical = {0:.6E} ; in cm\n".format(cors_asl_cm)
    sc += "\n"
    sc += "# parameters setting up the temporal observer configuration:\n"
    sc += "\n"
    sc += "TimeResolution = {0:.6E} ; in s\n".format(time_slice_duration_s)
    tb = time_boundaries
    sc += "AutomaticTimeBoundaries = {0:.6E}; ".format(
        tb["automatic_time_boundaries_s"]
    )
    sc += "0: off, x: automatic boundaries with width x in s\n"
    sc += "TimeLowerBoundary = {0:.6E} ; ".format(tb["time_lower_boundary_s"])
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "TimeUpperBoundary = {0:.6E} ; ".format(tb["time_upper_boundary_s"])
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "ResolutionReductionScale = 0 ; "
    sc += "0: off, x: decrease time resolution linearly every x cm in radius\n"
    sc += "\n"
    sc += "# parameters setting up the simulation functionality:\n"
    sc += "GroundLevelRefractiveIndex = 1.000292 ; "
    sc += "specify refractive index at 0 m asl\n"
    sc += "\n"
    sc += "# event information for Offline simulations:\n"
    sc += "\n"
    sc += "EventNumber = -1\n"
    sc += "RunNumber = -1\n"
    sc += "GPSSecs = 0\n"
    sc += "CoreEastingOffline = 0 ; in meters\n"
    sc += "CoreNorthingOffline = 0 ; in meters\n"
    sc += "CoreVerticalOffline = 0 ; in meters\n"
    sc += "RotationAngleForMagfieldDeclination = 0     ; in degrees\n"
    sc += "Comment = \n"
    sc += "\n"
    sc += "# event information for your convenience and backwards "
    sc += "compatibility with other software, these values are not used as "
    sc += "input parameters for the simulation:\n"
    sc += "\n"
    return sc


COREAS_TIME = 0
COREAS_NORTH_COMPONENT = 1
COREAS_WEST_COMPONENT = 2
COREAS_VERTICAL_COMPONENT = 3


def estimate_time_slice_duration_s(raw_antenna_time_series):
    return np.gradient(raw_antenna_time_series[0, :, 0]).mean()


def assert_same_time_slice_duration(
    raw_antenna_time_series, time_slice_duration
):
    for antenna in range(raw_antenna_time_series.shape[0]):
        time_slice_duration_this_antenna = np.gradient(
            raw_antenna_time_series[antenna, :, 0]
        )
        valid = (
            np.abs(time_slice_duration_this_antenna - time_slice_duration)
            < time_slice_duration / 10
        )
        assert np.all(valid)
    return time_slice_duration


def time_series_paths_in_numerical_order(path):
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


def read_raw_electric_fields(path):
    """
    Returns raw antenna responses in SI units (Volt/Meter).
    """
    all_time_series_paths = time_series_paths_in_numerical_order(path)
    antenna_responses = []
    CGS_to_SI = CGS_statVolt_per_cm_to_SI_Volt_per_meter
    for time_series_path in all_time_series_paths:
        raw = np.genfromtxt(time_series_path, dtype=np.float64)
        raw[:, COREAS_NORTH_COMPONENT] *= CGS_to_SI
        raw[:, COREAS_WEST_COMPONENT] *= CGS_to_SI
        raw[:, COREAS_VERTICAL_COMPONENT] *= CGS_to_SI
        antenna_responses.append(raw)
    return np.array(antenna_responses)


def make_electric_fields(raw_electric_fields):
    """
    Read time dependent electric field on reflector dish from event simulated
    at PATH. Returns dict containing all three components of the electric
    field and timing information. Electric Field will be returned in SI units.
    """
    raw = raw_electric_fields
    time_slice_duration_s = estimate_time_slice_duration_s(raw)
    assert_same_time_slice_duration(raw, time_slice_duration_s)

    global_start_time_s = raw[:, :, COREAS_TIME].min()

    start_time_offsets_s = raw[:, 0, COREAS_TIME] - global_start_time_s
    start_slice_offsets_s = np.round(
        start_time_offsets_s / time_slice_duration_s
    ).astype(np.int64)
    assert np.all(start_slice_offsets_s == 0)

    return {
        "time_slice_duration_s": time_slice_duration_s,
        "num_time_slices": raw.shape[1],
        "num_antennas": raw.shape[0],
        "electric_fields_V_per_m": raw[:, :, 1:4].astype(np.float32),
        "global_start_time_s": global_start_time_s,
    }
