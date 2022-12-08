import numpy as np
from . import lownoiseblock


def make_timing(
    lnb_local_oscillator_frequency=9.75e9,
    lnb_intermediate_frequency_start=950e6,
    lnb_intermediate_frequency_stop=1950e6,
    oversampling=6,
    time_window_duration=35e-9,
):
    assert lnb_local_oscillator_frequency > 0.0
    assert lnb_intermediate_frequency_start > 0.0
    assert lnb_intermediate_frequency_stop > 0.0
    assert lnb_intermediate_frequency_start < lnb_intermediate_frequency_stop
    assert oversampling > 0
    assert np.mod(oversampling, 1.0) < 1e-9

    tt = {}
    tt["oversampling"] = oversampling

    tt["lnb"] = {}
    tt["lnb"]["local_oscillator_frequency"] = lnb_local_oscillator_frequency
    tt["lnb"][
        "intermediate_frequency_start"
    ] = lnb_intermediate_frequency_start
    tt["lnb"]["intermediate_frequency_stop"] = lnb_intermediate_frequency_stop
    tt["lnb"]["bandwidth"] = (
        tt["lnb"]["intermediate_frequency_stop"]
        - tt["lnb"]["intermediate_frequency_start"]
    )

    tt["electric_fields"] = {}

    tt["electric_fields"]["frequency"] = (
        tt["lnb"]["local_oscillator_frequency"] * oversampling
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

    tt["start_time_probe"] = {}
    tt["start_time_probe"]["time_slice_duration"] = (
        10 * oversampling * tt["electric_fields"]["time_slice_duration"]
    )
    tt["start_time_probe"]["position"] = [0.0, 0.0, 0.0]
    tt["start_time_probe"]["time_lower_boundary"] = -7e3 * time_window_duration
    tt["start_time_probe"]["time_upper_boundary"] = 7e2 * time_window_duration

    return tt


def make_timing_from_lnb(
    lnb, oversampling=6, time_window_duration=35e-9,
):
    return make_timing(
        lnb_local_oscillator_frequency=lnb["local_oscillator_frequency"],
        lnb_intermediate_frequency_start=lnb["intermediate_frequency_start"],
        lnb_intermediate_frequency_stop=lnb["intermediate_frequency_stop"],
        oversampling=oversampling,
        time_window_duration=time_window_duration,
    )


def estimate_start_time_from_antnna_response(raw_time, raw_field_components):
    num_components = raw_field_components.shape[1]
    max_position_times = np.zeros(num_components)
    for component in range(num_components):
        first_slice = np.min(np.nonzero(raw_field_components[:, component]))
        max_position_times[component] = raw_time[first_slice]
    return np.median(max_position_times)


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
