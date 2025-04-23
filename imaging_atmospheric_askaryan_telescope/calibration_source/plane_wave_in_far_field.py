import numpy as np
import copy
import homogeneous_transformation as homtra

from .. import electric_fields
from .. import signal
from . import sine_wave_ramp


def make_config():
    c = {}
    c["__type__"] = "plane_wave"

    g = {}
    g["azimuth_rad"] = 0.0
    g["zenith_rad"] = 0.0
    g["polarization_angle_rad"] = 0.0
    g["distance_to_plane_defining_time_zero_m"] = 10e3
    c["geometry"] = g

    p = {}
    p["power_of_isotrop_and_point_like_emitter_W"] = 1.0
    p["distance_to_isotrop_and_point_like_emitter_m"] = 100e3
    c["power"] = p

    s = {}
    s["emission_frequency_Hz"] = 10e9
    s["emission_duration_s"] = 10e-9
    s["emission_ramp_up_duration_s"] = 2e-9
    s["emission_ramp_down_duration_s"] = 2e-9
    s["emission_overhead_duration_before_and_after_s"] = 1e-9
    c["sine_wave"] = s

    return c


def distance_between_plane_and_point(
    plane_support_vector,
    plane_normal_vector,
    point,
):
    """
    Returns the closest distance between a 3D plane and a point.
    """
    plane_support_vector = np.asarray(plane_support_vector)
    plane_normal_vector = np.asarray(plane_normal_vector)
    point = np.asarray(point)

    norm = np.linalg.norm
    dot = np.dot

    p = point
    q = plane_support_vector
    n = plane_normal_vector

    return norm(dot(p - q, n)) / norm(n)


def has_no_nan(x):
    """
    Returns True when array 'x' has no NaNs.
    """
    return np.all(np.logical_not(np.isnan(x)))


def make_civil_transformation(azimuth_rad, zenith_rad, polarization_angle_rad):
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


def make_power_setup(
    power_of_isotrop_and_point_like_emitter_W,
    distance_to_isotrop_and_point_like_emitter_m,
):
    assert power_of_isotrop_and_point_like_emitter_W >= 0.0
    assert distance_to_isotrop_and_point_like_emitter_m > 0.0
    vacuum_impedance_Ohm = signal.VACUUM_IMPEDANCE_OHM

    p = {}
    p["pointing_vector_magnitude_W_per_m2"] = (
        power_of_isotrop_and_point_like_emitter_W
        / (4.0 * np.pi * distance_to_isotrop_and_point_like_emitter_m**2.0)
    )

    p["electric_field_amplitue_V_per_m"] = np.sqrt(
        p["pointing_vector_magnitude_W_per_m2"] * vacuum_impedance_Ohm
    )
    return p


def make_geometry_setup(
    azimuth_rad,
    zenith_rad,
    polarization_angle_rad,
    distance_to_plane_defining_time_zero_m,
    antenna_position_vectors_in_asl_frame_m,
):
    assert not np.isnan(azimuth_rad)
    assert not np.isnan(zenith_rad)
    assert not np.isnan(polarization_angle_rad)
    assert not np.isnan(distance_to_plane_defining_time_zero_m)
    assert has_no_nan(antenna_position_vectors_in_asl_frame_m)

    g = {}
    g["azimuth_rad"] = azimuth_rad
    g["zenith_rad"] = zenith_rad
    g["polarization_angle_rad"] = polarization_angle_rad
    g["distance_to_plane_defining_time_zero_m"] = (
        distance_to_plane_defining_time_zero_m
    )

    g["antenna_position_vectors_in_asl_frame_m"] = np.asarray(
        antenna_position_vectors_in_asl_frame_m
    )

    g["Pointing_vector_in_source_frame"] = np.array([0.0, 0.0, 1.0])
    g["E_field_vector_in_source_frame"] = np.array([1.0, 0.0, 0.0])
    # B_field_vector_in_source_frame = np.array([0.0, 1.0, 0.0])

    g["homogeneous_transformation_from_source_frame_to_asl_frame"] = (
        homtra.compile(
            make_civil_transformation(
                azimuth_rad=g["azimuth_rad"],
                zenith_rad=g["zenith_rad"],
                polarization_angle_rad=g["polarization_angle_rad"],
            )
        )
    )

    g["Pointing_vector_in_asl_frame"] = homtra.transform_orientation(
        t=g["homogeneous_transformation_from_source_frame_to_asl_frame"],
        d=g["Pointing_vector_in_source_frame"],
    )
    g["E_field_vector_in_asl_frame"] = homtra.transform_orientation(
        t=g["homogeneous_transformation_from_source_frame_to_asl_frame"],
        d=g["E_field_vector_in_source_frame"],
    )

    g["plane_zero"] = {}
    g["plane_zero"]["normal_vector_in_asl_frame"] = copy.copy(
        g["Pointing_vector_in_asl_frame"]
    )
    g["plane_zero"]["support_vector_in_asl_frame"] = (
        +g["distance_to_plane_defining_time_zero_m"]
        * g["plane_zero"]["normal_vector_in_asl_frame"]
    )

    num_antennas = g["antenna_position_vectors_in_asl_frame_m"].shape[0]
    g["antenna_distances_to_plane_defining_time_zero_m"] = np.nan * np.ones(
        num_antennas
    )

    for a in range(num_antennas):
        g["antenna_distances_to_plane_defining_time_zero_m"][a] = (
            distance_between_plane_and_point(
                plane_support_vector=g["plane_zero"][
                    "support_vector_in_asl_frame"
                ],
                plane_normal_vector=g["plane_zero"][
                    "normal_vector_in_asl_frame"
                ],
                point=g["antenna_position_vectors_in_asl_frame_m"][a, :],
            )
        )

    g["min_antenna_distance_to_plane_defining_time_zero_m"] = np.min(
        g["antenna_distances_to_plane_defining_time_zero_m"]
    )
    g["max_antenna_distance_to_plane_defining_time_zero_m"] = np.max(
        g["antenna_distances_to_plane_defining_time_zero_m"]
    )

    return g


def plane_wave_in_far_field(
    geometry_setup,
    power_setup,
    sine_wave,
    time_slice_duration_s,
):
    assert sine_wave["emission_overhead_duration_before_and_after_s"] >= 0.0
    geom = geometry_setup
    pows = power_setup

    speed_of_light_m_per_s = signal.SPEED_OF_LIGHT_M_PER_S

    num_antennas = geom["antenna_position_vectors_in_asl_frame_m"].shape[0]

    time_duration_to_reach_closest_antenna_s = (
        geom["min_antenna_distance_to_plane_defining_time_zero_m"]
        / speed_of_light_m_per_s
    )
    time_duration_to_reach_furthest_antenna_s = (
        geom["max_antenna_distance_to_plane_defining_time_zero_m"]
        / speed_of_light_m_per_s
    )

    start_time_of_sampling_s = (
        time_duration_to_reach_closest_antenna_s
        - sine_wave["emission_ramp_up_duration_s"]
        - sine_wave["emission_overhead_duration_before_and_after_s"]
    )
    stop_time_of_sampling_s = (
        time_duration_to_reach_furthest_antenna_s
        + sine_wave["emission_duration_s"]
        + sine_wave["emission_ramp_down_duration_s"]
        + sine_wave["emission_overhead_duration_before_and_after_s"]
    )
    total_emission_duration_s = (
        stop_time_of_sampling_s - start_time_of_sampling_s
    )

    num_time_slices = int(
        np.ceil(
            (stop_time_of_sampling_s - start_time_of_sampling_s)
            / time_slice_duration_s
        )
    )

    E = electric_fields.init(
        time_slice_duration_s=time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_antennas=num_antennas,
        global_start_time_s=start_time_of_sampling_s,
    )

    assert 0.99 < np.linalg.norm(geom["E_field_vector_in_asl_frame"]) < 1.01
    assert pows["electric_field_amplitue_V_per_m"] > 0.0

    E_field_vector_in_asl_frame = Nx3_from_stacked_1x3(
        v=geom["E_field_vector_in_asl_frame"],
        size=E["num_time_slices"],
    )

    for a in range(num_antennas):
        time_duration_for_wave_to_reach_antenna_s = (
            geom["antenna_distances_to_plane_defining_time_zero_m"][a]
            / speed_of_light_m_per_s
        )

        sine_wave_amplitude = sine_wave_ramp.make_sine_wave_with_ramp_up_and_ramp_down(
            emission_frequency_Hz=sine_wave["emission_frequency_Hz"],
            emission_start_time_s=time_duration_for_wave_to_reach_antenna_s,
            emission_duration_s=sine_wave["emission_duration_s"],
            emission_ramp_up_duration_s=sine_wave[
                "emission_ramp_up_duration_s"
            ],
            emission_ramp_down_duration_s=sine_wave[
                "emission_ramp_down_duration_s"
            ],
            global_start_time_s=E["global_start_time_s"],
            time_slice_duration_s=E["time_slice_duration_s"],
            num_time_slices=E["num_time_slices"],
        )
        assert not np.all(sine_wave_amplitude == 0)

        E["electric_fields_V_per_m"][a] = copy.copy(
            E_field_vector_in_asl_frame
        )
        E["electric_fields_V_per_m"][a] = Nx3_multiply_elementwise_Nx1(
            nx3=E["electric_fields_V_per_m"][a],
            nx1=sine_wave_amplitude,
        )
        E["electric_fields_V_per_m"][a] = Nx3_multiply_elementwise_scalar(
            nx3=E["electric_fields_V_per_m"][a],
            scalar=pows["electric_field_amplitue_V_per_m"],
        )

    return E


def Nx3_from_stacked_1x3(v, size):
    assert len(v) == 3
    A = np.zeros(shape=(size, 3))
    A[:, 0] = v[0]
    A[:, 1] = v[1]
    A[:, 2] = v[2]
    return A


def Nx3_multiply_elementwise_Nx1(nx3, nx1):
    assert len(nx3.shape) == 2
    assert nx3.shape[1] == 3
    assert nx3.shape[0] == len(nx1)
    return nx3 * np.c_[nx1, nx1, nx1]


def Nx3_multiply_elementwise_scalar(nx3, scalar):
    assert len(nx3.shape) == 2
    assert nx3.shape[1] == 3
    _tmp_nx3 = scalar * np.ones(shape=nx3.shape)
    return nx3 * _tmp_nx3
