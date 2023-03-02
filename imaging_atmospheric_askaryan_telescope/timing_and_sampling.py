import numpy as np
from . import lownoiseblock
from . import signal


def make_timing_from_lnb(
    lnb,
    oversampling=6,
    readout_integrates_num_simulation_time_slices=234,
    time_window_duration=35e-9,
):
    assert lnb["local_oscillator_frequency"] > 0.0
    assert lnb["intermediate_frequency_start"] > 0.0
    assert lnb["intermediate_frequency_stop"] > 0.0
    assert lnb["intermediate_frequency_start"] < lnb["intermediate_frequency_stop"]
    assert oversampling > 0
    assert readout_integrates_num_simulation_time_slices > 0
    assert np.mod(oversampling, 1.0) < 1e-9

    tt = {}
    tt["oversampling"] = oversampling

    tt["electric_fields"] = {}

    tt["electric_fields"]["frequency"] = (
        lnb["local_oscillator_frequency"] * oversampling
    )
    tt["electric_fields"]["time_slice_duration"] = (
        1.0 / tt["electric_fields"]["frequency"]
    )

    tt["electric_fields"]["mirror"] = {}
    tt["electric_fields"]["mirror"]["time_window_duration"] = (
        1.1 * time_window_duration
    )
    tt["electric_fields"]["mirror"][
        "warm_up_fraction_wrt_to_start_time_probe"
    ] = 0.06

    tt["electric_fields"]["sensor"] = {}
    tt["electric_fields"]["sensor"]["num_time_slices"] = int(
        np.ceil(
            time_window_duration / tt["electric_fields"]["time_slice_duration"]
        )
    )

    """
    From the CoReas Manual:

    TimeLowerBoundary
    -----------------
    Sets a global lower bound for the time window to be calculated. (Only applicable
    if AutomaticTimeBoundaries=0 and not recommended, setting of Automatic-
    TimeBoundaries is preferred.) Value is in seconds, where 0 denotes the time
    when an imaginary leading particle propagating at the speed of light hits the
    specified shower core.
    """
    OVERHEAD_DISTANCE = 10e4
    TIME_TO_TRAVEL_OVERHEAD = OVERHEAD_DISTANCE / signal.SPEED_OF_LIGHT

    tt["start_time_probe"] = {}
    tt["start_time_probe"]["time_slice_duration"] = (
        10 * oversampling * tt["electric_fields"]["time_slice_duration"]
    )
    tt["start_time_probe"]["position"] = [0.0, 0.0, 0.0]
    tt["start_time_probe"]["time_lower_boundary"] = -TIME_TO_TRAVEL_OVERHEAD
    tt["start_time_probe"]["time_upper_boundary"] = TIME_TO_TRAVEL_OVERHEAD

    tt["readout"] = {}
    tt["readout"][
        "integrates_num_simulation_time_slices"
    ] = readout_integrates_num_simulation_time_slices
    tt["readout"]["time_slice_duration"] = (
        tt["electric_fields"]["time_slice_duration"]
        * tt["readout"]["integrates_num_simulation_time_slices"]
    )
    tt["readout"]["frequency"] = 1.0 / tt["readout"]["time_slice_duration"]

    return tt


def estimate_start_time_from_electric_fields(electric_fields):
    e = electric_fields
    first_slices = []
    for ant in range(e["num_antennas"]):
        for dim in range(3):
            first_slice = np.min(np.nonzero(e["electric_fields"][ant, :, dim]))
            first_slices.append(first_slice)

    start_slice = np.median(first_slices)
    start_time_relative = start_slice * e["time_slice_duration"]
    start_time = start_time_relative + e["global_start_time"]
    return start_time


def make_time_window_bounds(
    start_time,
    time_window_duration,
    fraction_of_time_window_to_be_warm_up_time,
):
    assert time_window_duration > 0.0
    assert fraction_of_time_window_to_be_warm_up_time >= 0.0

    f = fraction_of_time_window_to_be_warm_up_time
    time_lower_boundary = start_time - f * time_window_duration
    time_upper_boundary = start_time + (1 - f) * time_window_duration

    return time_lower_boundary, time_upper_boundary
