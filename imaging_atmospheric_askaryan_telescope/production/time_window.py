# Copyright 2017 Sebastian A. Mueller
import numpy as np


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
