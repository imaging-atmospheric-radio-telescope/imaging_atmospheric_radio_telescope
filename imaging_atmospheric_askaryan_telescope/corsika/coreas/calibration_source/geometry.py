import spherical_coordinates
import homogeneous_transformation as homtra
import numpy as np
import copy


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


def has_no_nan(x):
    return np.all(np.logical_not(np.isnan(x)))


def _make_geometry(
    azimuth_rad,
    zenith_rad,
    polarization_angle_rad,
    distance_to_plane_defining_time_zero_m,
    core_position_vector_in_asl_frame_m,
    antenna_position_vectors_in_asl_frame_m,
):
    assert not np.isnan(azimuth_rad)
    assert not np.isnan(zenith_rad)
    assert not np.isnan(polarization_angle_rad)
    assert not np.isnan(distance_to_plane_defining_time_zero_m)
    assert has_no_nan(core_position_vector_in_asl_frame_m)
    assert has_no_nan(antenna_position_vectors_in_asl_frame_m)

    g = {}
    g["azimuth_rad"] = azimuth_rad
    g["zenith_rad"] = zenith_rad
    g["polarization_angle_rad"] = polarization_angle_rad
    g["distance_to_plane_defining_time_zero_m"] = (
        distance_to_plane_defining_time_zero_m
    )

    g["core_position_vector_in_asl_frame_m"] = np.asarray(
        core_position_vector_in_asl_frame_m
    )
    g["antenna_position_vectors_in_asl_frame_m"] = np.asarray(
        antenna_position_vectors_in_asl_frame_m
    )

    g["Pointing_vector_in_source_frame"] = np.array([0.0, 0.0, 1.0])
    g["E_field_vector_in_source_frame"] = np.array([1.0, 0.0, 0.0])
    # B_field_vector_in_source_frame = np.array([0.0, 1.0, 0.0])

    g["homogeneous_transformation_from_source_frame_to_asl_frame"] = (
        homtra.compile(
            make_civil_transformation_for_plane_wave(
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
        g["core_position_vector_in_asl_frame_m"]
        + g["distance_to_plane_defining_time_zero_m"]
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
