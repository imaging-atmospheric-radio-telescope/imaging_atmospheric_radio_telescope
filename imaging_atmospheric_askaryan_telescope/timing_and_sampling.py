import numpy as np
from . import lownoiseblock
from . import signal


def default_config():
    """
    Default config for timing and sampling.
    """
    return {
        "oversampling": 6,
        "time_window_duration_s": 3.5e-08,
        "readout_sampling_rate_per_s": 250e6,
    }


def is_close_to_integer(x, epsilon=1e-9):
    return np.mod(x, 1.0) < epsilon


def make_timing_from_lnb(
    lnb,
    oversampling,
    readout_sampling_rate_per_s,
    time_window_duration_s,
):
    assert lnb["local_oscillator_frequency_Hz"] > 0.0
    assert lnb["intermediate_frequency_start_Hz"] > 0.0
    assert lnb["intermediate_frequency_stop_Hz"] > 0.0
    assert (
        lnb["intermediate_frequency_start_Hz"]
        < lnb["intermediate_frequency_stop_Hz"]
    )
    assert oversampling > 0
    assert readout_sampling_rate_per_s > 0
    assert is_close_to_integer(oversampling)
    oversampling = int(np.round(oversampling))

    tt = {}
    tt["oversampling"] = oversampling

    tt["electric_fields"] = {}

    _, lnb_input_stop_Hz = lownoiseblock.input_frequency_start_stop_Hz(lnb=lnb)

    tt["electric_fields"]["sampling_rate_per_s"] = (
        lnb_input_stop_Hz * oversampling
    )
    tt["electric_fields"]["time_slice_duration_s"] = (
        1.0 / tt["electric_fields"]["sampling_rate_per_s"]
    )

    tt["electric_fields"]["mirror"] = {}
    tt["electric_fields"]["mirror"]["time_window_duration_s"] = (
        1.1 * time_window_duration_s
    )
    tt["electric_fields"]["mirror"][
        "warm_up_fraction_wrt_to_start_time_probe"
    ] = 0.06

    tt["electric_fields"]["sensor"] = {}
    tt["electric_fields"]["sensor"]["num_time_slices"] = int(
        np.ceil(
            time_window_duration_s
            / tt["electric_fields"]["time_slice_duration_s"]
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
    OVERHEAD_DISTANCE_M = 10e4
    TIME_TO_TRAVEL_OVERHEAD_S = (
        OVERHEAD_DISTANCE_M / signal.SPEED_OF_LIGHT_M_PER_S
    )

    tt["start_time_probe"] = {}
    tt["start_time_probe"]["time_slice_duration_s"] = (
        10 * oversampling * tt["electric_fields"]["time_slice_duration_s"]
    )
    tt["start_time_probe"]["position_m"] = [0.0, 0.0, 0.0]
    tt["start_time_probe"][
        "time_lower_boundary_s"
    ] = -TIME_TO_TRAVEL_OVERHEAD_S
    tt["start_time_probe"]["time_upper_boundary_s"] = TIME_TO_TRAVEL_OVERHEAD_S

    tt["readout"] = {}
    tt["readout"]["integrates_num_simulation_time_slices"] = int(
        np.round(
            tt["electric_fields"]["sampling_rate_per_s"]
            / readout_sampling_rate_per_s
        )
    )

    tt["readout"]["time_slice_duration_s"] = (
        tt["electric_fields"]["time_slice_duration_s"]
        * tt["readout"]["integrates_num_simulation_time_slices"]
    )
    tt["readout"]["sampling_rate_per_s"] = (
        1.0 / tt["readout"]["time_slice_duration_s"]
    )

    return tt


def make_time_window_bounds(
    start_time_s,
    time_window_duration_s,
    fraction_of_time_window_to_be_warm_up_time,
):
    assert time_window_duration_s > 0.0
    assert fraction_of_time_window_to_be_warm_up_time >= 0.0

    f = fraction_of_time_window_to_be_warm_up_time
    time_lower_boundary_s = start_time_s - f * time_window_duration_s
    time_upper_boundary_s = start_time_s + (1 - f) * time_window_duration_s

    return time_lower_boundary_s, time_upper_boundary_s
