from imaging_atmospheric_askaryan_telescope.corsika.coreas import (
    calibration_source,
)
import numpy as np


def assert_amplitude_at(actual_amplitudes, s, desired_amplitude, decimal=6):
    if 0 <= s < len(actual_amplitudes):
        np.testing.assert_almost_equal(
            actual=actual_amplitudes[s],
            desired=desired_amplitude,
            decimal=decimal,
        )


def assert_make_sine_wave_with_ramp_up_and_down(A, args):
    assert len(A) == args["num_time_slices"]

    t_start_of_ramp_up_s = (
        args["emission_start_time_s"] - args["emission_ramp_up_duration_s"]
    )
    t_start_of_emission_s = args["emission_start_time_s"]
    t_start_of_ramp_down_s = (
        args["emission_start_time_s"] + args["emission_duration_s"]
    )
    t_end_of_ramp_down_s = (
        t_start_of_ramp_down_s + args["emission_ramp_down_duration_s"]
    )

    s_start_of_ramp_up = calibration_source.time_to_slice(
        t_start_of_ramp_up_s, args["time_slice_duration_s"]
    )
    s_start_of_emission = calibration_source.time_to_slice(
        t_start_of_emission_s, args["time_slice_duration_s"]
    )
    s_start_of_ramp_down = calibration_source.time_to_slice(
        t_start_of_ramp_down_s, args["time_slice_duration_s"]
    )
    s_end_of_ramp_down = calibration_source.time_to_slice(
        t_end_of_ramp_down_s, args["time_slice_duration_s"]
    )
    s_period = calibration_source.time_to_slice(
        1.0 / args["emission_frequency_Hz"], args["time_slice_duration_s"]
    )

    assert s_start_of_ramp_up < s_start_of_emission
    assert s_start_of_emission < s_start_of_ramp_down
    assert s_start_of_ramp_down < s_end_of_ramp_down

    # test zero before ramp up
    for s in np.arange(0, s_start_of_ramp_up):
        if 0 <= s < args["num_time_slices"]:
            assert A[s] == 0.0

    # test ramp up
    for s in np.arange(s_start_of_ramp_up, s_start_of_emission):
        weight = (s - s_start_of_ramp_up) / (
            s_start_of_emission - s_start_of_ramp_up
        )
        if 0 <= s < args["num_time_slices"]:
            assert -weight <= A[s] <= weight

    # test emission amplitude
    for s in np.arange(s_start_of_emission, s_start_of_ramp_down):
        if 0 <= s < args["num_time_slices"]:
            assert -1.0 <= A[s] <= 1.0

    # test phase of emission is where it is expected

    assert_amplitude_at(
        actual_amplitudes=A,
        s=s_start_of_emission,
        desired_amplitude=0.0,
        decimal=1,
    )

    assert_amplitude_at(
        actual_amplitudes=A,
        s=s_start_of_emission + s_period // 4,
        desired_amplitude=1.0,
        decimal=1,
    )

    assert_amplitude_at(
        actual_amplitudes=A,
        s=s_start_of_emission + s_period // 2,
        desired_amplitude=0.0,
        decimal=1,
    )

    assert_amplitude_at(
        actual_amplitudes=A,
        s=s_start_of_emission + s_period,
        desired_amplitude=0.0,
        decimal=1,
    )

    # test ramp down
    for s in np.arange(s_start_of_ramp_down, s_end_of_ramp_down):
        weight = 1.0 - (s - s_start_of_ramp_down) / (
            s_end_of_ramp_down - s_start_of_ramp_down
        )
        if 0 <= s < args["num_time_slices"]:
            assert -weight <= A[s] <= weight

    # test zero after ramp down
    for s in np.arange(s_end_of_ramp_down, args["num_time_slices"]):
        if 0 <= s < args["num_time_slices"]:
            assert A[s] == 0.0


def make_example_args():
    return {
        "emission_frequency_Hz": 3e4,
        "emission_start_time_s": 0.25,
        "emission_duration_s": 0.5,
        "emission_ramp_up_duration_s": 50e-3,
        "emission_ramp_down_duration_s": 50e-3,
        "time_slice_duration_s": 1e-6,
        "num_time_slices": 1_000_000,
    }


def test_example():
    args = make_example_args()
    A = calibration_source.make_sine_wave_with_ramp_up_and_ramp_down(**args)
    assert_make_sine_wave_with_ramp_up_and_down(A=A, args=args)


def test_ramp_up_is_behind_last_time_slice():
    args = make_example_args()
    args["emission_start_time_s"] = 2.0
    A = calibration_source.make_sine_wave_with_ramp_up_and_ramp_down(**args)
    assert_make_sine_wave_with_ramp_up_and_down(A=A, args=args)


def test_end_of_ramp_down_is_before_first_time_slice():
    args = make_example_args()
    args["emission_start_time_s"] = -2.0
    A = calibration_source.make_sine_wave_with_ramp_up_and_ramp_down(**args)
    assert_make_sine_wave_with_ramp_up_and_down(A=A, args=args)
