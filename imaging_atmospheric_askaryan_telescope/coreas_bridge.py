# Copyright 2017 Sebastian A. Mueller
import numpy as np
import glob
import os

COREAS_TIME = 0
COREAS_NORTH_COMPONENT = 1
COREAS_WEST_COMPONENT = 2
COREAS_VERTICAL_COMPONENT = 3


def estimate_time_slice_duration(raw_antenna_time_series):
    return np.gradient(raw_antenna_time_series[0, :, 0]).mean()


def assert_same_time_slice_duration(
    raw_antenna_time_series,
    time_slice_duration
):
    for antenna in range(raw_antenna_time_series.shape[0]):
        time_slice_duration_this_antenna = np.gradient(
            raw_antenna_time_series[antenna, :, 0]
        )
        valid = (
            np.abs(time_slice_duration_this_antenna - time_slice_duration) <
            time_slice_duration / 100
        )
        assert np.all(valid)
    return time_slice_duration


def time_series_paths_in_numerical_order(path):
    all_time_series_paths = glob.glob(os.path.join(path, '*.dat'))
    antenna_indices = []
    for time_series_path in all_time_series_paths:
        basename = os.path.basename(time_series_path)
        antenna_index = int(basename[20:26])
        antenna_indices.append(antenna_index)
    antenna_indices = np.array(antenna_indices)
    order = np.argsort(antenna_indices)
    all_time_series_paths = [all_time_series_paths[i] for i in order]
    return all_time_series_paths


CGS_statVolt_per_cm_to_SI_Volt_per_meter = 2.99792458e4


def read_all_raw_time_series(path):
    '''
    Returns raw antenna responses in SI units (Volt/Meter).
    '''
    all_time_series_paths = time_series_paths_in_numerical_order(path)
    antenna_responses = []
    CGS_to_SI = CGS_statVolt_per_cm_to_SI_Volt_per_meter
    for time_series_path in all_time_series_paths:
        time_series = np.genfromtxt(time_series_path, dtype=np.float32)
        time_series[:, COREAS_NORTH_COMPONENT] *= CGS_to_SI
        time_series[:, COREAS_WEST_COMPONENT] *= CGS_to_SI
        time_series[:, COREAS_VERTICAL_COMPONENT] *= CGS_to_SI
        antenna_responses.append(time_series)
    return np.array(antenna_responses)


def read_electric_field_on_imaging_reflector(path):
    '''
    Read time dependent electric field on reflector dish from event simulated at PATH.
    Returns dict containing all three components of the electric field and timing information.
    Electric Field will be returned in SI units.
    '''
    raw = read_all_raw_time_series(path)
    time_slice_duration = estimate_time_slice_duration(raw)
    assert_same_time_slice_duration(raw, time_slice_duration)

    global_start_time = raw[:, :, COREAS_TIME].min()
    global_end_time = raw[:, :, COREAS_TIME].max()

    antenna_start_time_offsets = raw[:, 0, COREAS_TIME] - global_start_time
    antenna_start_slice_offsets = np.round(
        antenna_start_time_offsets / time_slice_duration
    ).astype(np.int64)

    return {
        'time_slice_duration': time_slice_duration,
        'global_start_time': global_start_time,
        'global_end_time': global_end_time,
        'antenna_start_time_offsets': antenna_start_time_offsets,
        'antenna_start_slice_offsets': antenna_start_slice_offsets,
        'north_component': raw[:, :, COREAS_NORTH_COMPONENT],
        'west_component': raw[:, :, COREAS_WEST_COMPONENT],
        'vertical_component': raw[:, :, COREAS_VERTICAL_COMPONENT],
        'number_time_slices': raw.shape[1],
    }
