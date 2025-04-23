from imaging_atmospheric_askaryan_telescope import (
    calibration_source,
)
from imaging_atmospheric_askaryan_telescope import (
    signal,
)
import numpy as np


def make_example_args():
    return {
        "azimuth_rad": 0,
        "zenith_rad": 0,
        "polarization_angle_rad": 0,
        "power_of_isotrop_and_point_like_emitter_W": 0,
        "distance_to_isotrop_and_point_like_emitter_m": 0,
        "emission_frequency_Hz": 0,
        "emission_duration_s": 0,
        "emission_ramp_up_duration_s": 0,
        "emission_ramp_down_duration_s": 0,
        "distance_to_define_time_zero": 0,
        "antenna_positions_asl_m": 0,
        "time_slice_duration_s": 0,
        "core_position_on_observation_level_north_m": 0,
        "core_position_on_observation_level_west_m": 0,
        "core_position_on_observation_level_asl_m": 0,
    }


def test_distance_between_plane_and_point():
    for z in np.linspace(-5, 6, 101):
        d = calibration_source.plane_wave_in_far_field.distance_between_plane_and_point(
            plane_support_vector=[0, 0, z],
            plane_normal_vector=[0, 0, 1],
            point=[0, 0, 0],
        )
        np.testing.assert_almost_equal(actual=d, desired=np.abs(z))


def test_geometry_setup_most_simple_case():
    geom = calibration_source.plane_wave_in_far_field.make_geometry_setup(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        polarization_angle_rad=0.0,
        distance_to_plane_defining_time_zero_m=1_000,
        core_position_vector_in_asl_frame_m=[0, 0, 0],
        antenna_position_vectors_in_asl_frame_m=[[0, 0, 0]],
    )

    np.testing.assert_array_almost_equal(
        actual=geom["plane_zero"]["support_vector_in_asl_frame"],
        desired=[0, 0, 1_000],
    )
    np.testing.assert_array_almost_equal(
        actual=geom["plane_zero"]["normal_vector_in_asl_frame"],
        desired=[0, 0, 1],
    )
    np.testing.assert_almost_equal(
        actual=geom["antenna_distances_to_plane_defining_time_zero_m"],
        desired=1_000,
    )


def test_power_setup():
    pows = calibration_source.plane_wave_in_far_field.make_power_setup(
        power_of_isotrop_and_point_like_emitter_W=3.0,
        distance_to_isotrop_and_point_like_emitter_m=100.0,
    )
    assert pows["pointing_vector_magnitude_W_per_m2"] == 3 / (
        4 * np.pi * 100**2
    )
    assert pows["electric_field_amplitue_V_per_m"] == np.sqrt(
        pows["pointing_vector_magnitude_W_per_m2"] * signal.VACUUM_IMPEDANCE
    )
