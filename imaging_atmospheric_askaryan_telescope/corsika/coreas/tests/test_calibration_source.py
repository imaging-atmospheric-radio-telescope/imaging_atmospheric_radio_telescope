from imaging_atmospheric_askaryan_telescope.corsika.coreas import (
    calibration_source,
)
import numpy as np


def test_make_sine_wave_with_ramp_up_and_down():

    args = {
        "emission_frequency_Hz": 3e4,
        "emission_start_time_s": 0.25,
        "emission_duration_s": 0.5,
        "emission_ramp_up_duration_s": 50e-3,
        "emission_ramp_down_duration_s": 50e-3,
        "time_slice_duration_s": 1e-6,
        "num_time_slices": 1_000_000,
    }

    A = calibration_source.make_sine_wave_with_ramp_up_and_ramp_down(**args)

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

    s_start_of_ramp_up = calibration_source.to_time_slice(
        t_start_of_ramp_up_s, args["time_slice_duration_s"]
    )
    s_start_of_emission = calibration_source.to_time_slice(
        t_start_of_emission_s, args["time_slice_duration_s"]
    )
    s_start_of_ramp_down = calibration_source.to_time_slice(
        t_start_of_ramp_down_s, args["time_slice_duration_s"]
    )
    s_end_of_ramp_down = calibration_source.to_time_slice(
        t_end_of_ramp_down_s, args["time_slice_duration_s"]
    )
    s_period = calibration_source.to_time_slice(
        1.0 / args["emission_frequency_Hz"], args["time_slice_duration_s"]
    )

    assert 0 < s_start_of_ramp_up
    assert s_start_of_ramp_up < s_start_of_emission
    assert s_start_of_emission < s_start_of_ramp_down
    assert s_start_of_ramp_down < s_end_of_ramp_down
    assert s_end_of_ramp_down < args["num_time_slices"]

    # test zero before ramp up
    for s in np.arange(0, s_start_of_ramp_up):
        assert A[s] == 0.0

    # test ramp up
    for s in np.arange(s_start_of_ramp_up, s_start_of_emission):
        weight = (s - s_start_of_ramp_up) / (
            s_start_of_emission - s_start_of_ramp_up
        )
        assert -weight <= A[s] <= weight

    # test emission
    for s in np.arange(s_start_of_emission, s_start_of_ramp_down):
        assert -1.0 <= A[s] <= 1.0

    # test phase of emission
    np.testing.assert_almost_equal(actual=A[s_start_of_emission], desired=0.0)
    np.testing.assert_almost_equal(
        actual=A[s_start_of_emission + s_period // 4], desired=1.0, decimal=1
    )
    np.testing.assert_almost_equal(
        actual=A[s_start_of_emission + s_period // 2], desired=0.0, decimal=1
    )
    np.testing.assert_almost_equal(
        actual=A[s_start_of_emission + s_period], desired=0.0, decimal=1
    )

    # test ramp down
    for s in np.arange(s_start_of_ramp_down, s_end_of_ramp_down):
        weight = 1.0 - (s - s_start_of_ramp_down) / (
            s_end_of_ramp_down - s_start_of_ramp_down
        )
        assert -weight <= A[s] <= weight

    # test zero after ramp down
    for s in np.arange(s_end_of_ramp_down, args["num_time_slices"]):
        assert A[s] == 0.0
