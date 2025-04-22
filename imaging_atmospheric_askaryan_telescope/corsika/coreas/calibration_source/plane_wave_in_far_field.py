import collections

import numpy as np
import copy
from ... import electric_fields
from ... import signal


def point_source_emitting_spherical_wave_in_far_field(
    azimuth_rad,
    zenith_rad,
    polarization_angle_rad,
    power_of_isotrop_and_point_like_emitter_W,
    distance_to_isotrop_and_point_like_emitter_m,
    emission_frequency_Hz,
    emission_duration_s,
    emission_ramp_up_duration_s,
    emission_ramp_down_duration_s,
    distance_to_define_time_zero,
    antenna_positions_asl_m,
    time_slice_duration_s,
    core_position_on_observation_level_north_m,
    core_position_on_observation_level_west_m,
    core_position_on_observation_level_asl_m,
):
    vacuum_impedance_Ohm = signal.VACUUM_IMPEDANCE
    speed_of_light_m_per_s = signal.SPEED_OF_LIGHT

    assert power_of_isotrop_and_point_like_emitter_W >= 0.0
    assert distance_to_isotrop_and_point_like_emitter_m > 0.0

    total_emission_duration_s = (
        emission_ramp_up_duration_s
        + emission_duration_s
        + emission_ramp_down_duration_s
    )

    num_time_slices = int(
        np.ceil(total_emission_duration_s / time_slice_duration_s)
    )

    num_antennas = antenna_positions_asl_m.shape[0]

    plane_to_define_time_zero_normal_vector = source_pointing_direction_vector

    plane_to_define_time_zero_support_vector = (
        core_position
        + distance_to_define_time_zero * source_pointing_direction_vector
    )

    pointing_vector_magnitude_W_per_m2 = (
        power_of_isotrop_and_point_like_emitter_W
        / (4.0 * np.pi * distance_to_isotrop_and_point_like_emitter_m**2.0)
    )

    electric_field_amplitue_V_per_m = np.sqrt(
        pointing_vector_magnitude_W_per_m2 * vacuum_impedance_Ohm
    )

    antenna_distances_to_plane_defining_time_zero_m = np.nan * np.ones(
        num_antennas
    )

    for a in range(num_antennas):
        antenna_distances_to_plane_defining_time_zero_m[a] = (
            distance_between_plane_and_point(
                plane_support_vector=plane_to_define_time_zero_support_vector,
                plane_normal_vector=plane_to_define_time_zero_normal_vector,
                point=antenna_positions_asl_m[a, :],
            )
        )

    min_antenna_distance_to_plane_defining_time_zero_m = np.min(
        antenna_distances_to_plane_defining_time_zero_m
    )
    max_antenna_distance_to_plane_defining_time_zero_m = np.max(
        antenna_distances_to_plane_defining_time_zero_m
    )

    time_duration_to_reach_closest_antenna_s = (
        min_antenna_distance_to_plane_defining_time_zero_m
        / speed_of_light_m_per_s
    )
    time_duration_to_reach_furthest_antenna_s = (
        max_antenna_distance_to_plane_defining_time_zero_m
        / speed_of_light_m_per_s
    )

    start_time_of_sampling_s = time_duration_to_reach_closest_antenna_s
    stop_time_of_sampling_s = (
        time_duration_to_reach_furthest_antenna_s + total_emission_duration_s
    )

    time_s = np.arange(
        start_time_of_sampling_s,
        stop_time_of_sampling_s,
        time_slice_duration_s,
    )

    E = electric_fields.init(
        time_slice_duration_s=time_slice_duration_s,
        num_time_slices=len(time_s),
        num_antennas=num_antennas,
        global_start_time_s=start_time_of_sampling_s,
    )

    for a in range(num_antennas):
        time_duration_for_wave_to_reach_antenna_s = (
            antenna_distances_to_plane_defining_time_zero_m[a]
            / speed_of_light_m_per_s
        )

        global_time_when_wave_reaches_antenna_s = (
            global_start_time_s + time_duration_for_wave_to_reach_antenna_s
        )

        amplitude = make_sine_wave_with_ramp_up_and_ramp_down(
            emission_frequency_Hz=emission_frequency_Hz,
            emission_start_time_s=global_time_when_wave_reaches_antenna_s,
            emission_duration_s=emission_duration_s,
            emission_ramp_up_duration_s=emission_ramp_up_duration_s,
            emission_ramp_down_duration_s=emission_ramp_down_duration_s,
            global_start_time_s=E["global_start_time_s"],
            time_slice_duration_s=E["time_slice_duration_s"],
            num_time_slices=E["num_time_slices"],
        )

        E["electric_fields_V_per_m"][a] = (
            amplitude * electric_field_amplitue_V_per_m * source_E_direction
        )

    return E
