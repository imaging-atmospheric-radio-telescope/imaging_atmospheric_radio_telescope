import collections
import spherical_coordinates
import homogeneous_transformation as homtra
import numpy as np
from ... import electric_fields


def distance_between_plane_and_point(
    plane_support_vector,
    plane_normal_vector,
    point,
):
    """
    Returns the closest distance between a 3D plane and a point.
    """
    norm = np.linalg.norm
    dot = np.dot

    p = point
    q = plane_support_vector
    n = plane_normal_vector

    return norm(dot(p - q, n)) / norm(n)


E_FIELD_DIRECTION = np.array([1.0, 0.0, 0.0])
B_FIELD_DIRECTION = np.array([0.0, 1.0, 0.0])
PROPAGATION_DIRECTION = np.array([0.0, 0.0, 1.0])


def make_civil_transformation_for_plane_wave(
    azimuth_rad, zenith_rad, polarization_angle_rad
):
    """
    Defining the 3D transformation of the plane wave.

    In its own frame the plane wave is:

    Electric field swings along the x axis
    Magnetic field swings along the y axis
    propagation is along the z axis

    Parameters
    ----------
    azimuth_rad : float
        Point source azimuth angle w.r.t. observation level.
    zenith_rad : float
        Point source zenith distance angle w.r.t. observation level.
    polarization_angle_rad : float
        Angle of electric field axis with respect to its own reference frame
        (x axis).
    """
    rot = {
        "repr": "tait_bryan",
        "xyz_deg": np.array(
            [
                np.rad2deg(polarization_angle_rad),
                np.rad2deg(-zenith_rad),
                np.rad2deg(-azimuth_rad),
            ]
        ),
    }
    zero = np.array([0, 0, 0])  # we only want to rotate.

    return {"pos": zero, "rot": rot}


def plane_wave(
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
    vacuum_impedance_Ohm = 120.0 * np.pi
    speed_of_light_m_per_s = 299792458.0
    source_pointing_direction_in_source_frame = np.array([0.0, 0.0, 1.0])
    source_E_direction_in_source_frame = np.array([1.0, 0.0, 0.0])
    # source_B_direction_in_source_frame = np.array([0.0, 1.0, 0.0])

    assert power_of_isotrop_and_point_like_emitter_W >= 0.0
    assert distance_to_isotrop_and_point_like_emitter_m > 0.0
    assert emission_frequency_Hz > 0.0
    assert emission_duration_s >= 0.0
    assert emission_ramp_up_duration_s >= 0.0
    assert emission_ramp_down_duration_s >= 0.0
    assert time_slice_duration_s > 0.0

    core_position = np.array(
        [
            core_position_on_observation_level_north_m,
            core_position_on_observation_level_west_m,
            core_position_on_observation_level_asl_m,
        ]
    )

    T_planewave_to_observation_level = homtra.compile(
        make_civil_transformation_for_plane_wave(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            polarization_angle_rad=polarization_angle_rad,
        )
    )

    source_pointing_direction = homtra.transform_orientation(
        t=T_planewave_to_observation_level,
        d=source_pointing_direction_in_source_frame,
    )
    source_E_direction = homtra.transform_orientation(
        t=T_planewave_to_observation_level,
        d=source_E_direction_in_source_frame,
    )

    total_emission_duration_s = (
        emission_ramp_up_duration_s
        + emission_duration_s
        + emission_ramp_down_duration_s
    )

    num_time_slices = int(
        np.ceil(total_emission_duration_s / time_slice_duration_s)
    )

    num_antennas = antenna_positions_asl_m.shape[0]

    plane_to_define_time_zero_normal_vector = source_pointing_direction

    plane_to_define_time_zero_support_vector = (
        core_position
        + distance_to_define_time_zero * source_pointing_direction
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
        d_m = antenna_distances_to_plane_defining_time_zero_m[a]

        E["electric_fields_V_per_m"][a] = _e_field

    return E


def time_to_slice(t, dt):
    return int(np.round(t / dt))


def make_sine_wave_with_ramp_up_and_ramp_down(
    emission_frequency_Hz,
    emission_start_time_s,
    emission_duration_s,
    emission_ramp_up_duration_s,
    emission_ramp_down_duration_s,
    time_slice_duration_s,
    num_time_slices,
):
    N = num_time_slices
    dt = time_slice_duration_s

    t_sine = emission_start_time_s
    t_up = t_sine - emission_ramp_up_duration_s
    t_down = t_sine + emission_duration_s
    t_end = t_down + emission_ramp_down_duration_s

    # in slices

    s_up = time_to_slice(t=t_up, dt=dt)
    s_sine = time_to_slice(t=t_sine, dt=dt)
    s_down = time_to_slice(t=t_down, dt=dt)
    s_end = time_to_slice(t=t_end, dt=dt)

    TAU = 2.0 * np.pi

    A = np.zeros(N, dtype=float)
    t = np.linspace(0.0, N * dt, N, endpoint=False)

    A = np.sin((t - emission_start_time_s) * emission_frequency_Hz * TAU)

    # zeros before s_up
    # -----------------
    for s in np.arange(0, min([N, s_up])):
        if 0 <= s < N:
            A[s] = 0.0

    # ramp up
    # -------
    N_ramp_up = s_sine - s_up
    for s in np.arange(s_up, s_sine):
        weight = (s - s_up) / N_ramp_up
        if 0 <= s < N:
            A[s] = A[s] * weight

    # the sine itself
    # ---------------
    N_ramp_down = s_end - s_down
    for s in np.arange(s_down, s_end):
        weight = 1.0 - ((s - s_down) / N_ramp_down)
        if 0 <= s < N:
            A[s] = A[s] * weight

    # zeros after end
    # ---------------
    for s in np.arange(s_end, N):
        if 0 <= s < N:
            A[s] = 0.0

    return A
