# Copyright 2017 Sebastian A. Mueller
import numpy as np
import scipy
from scipy import spatial
import scipy.spatial.distance
import os
from . import signal


def make_parabola_surface_height_m(
    distance_to_optical_axis_m, focal_length_m,
):
    """
    Parameters
    ----------
    distance_to_optical_axis_m : float
        The distance to the parabola's optical axis for where the height of
        the parabola is estimated.
    focal_length_m : float
        The parabola's focal-length.
    """
    z = 1 / (4.0 * focal_length_m) * distance_to_optical_axis_m ** 2
    return z


def make_probe_positions(
    random_seed,
    focal_length_m,
    outer_radius_m,
    inner_radius_m,
    scatter_center_areal_density_per_m2,
):
    """
    Returns the randomly drawn positions of scatter-centers on a parabolic
    imaging reflector. The x-, y-positions can be limited in an anulus with
    an innner, and outer radius. The z-position is computed from the
    focal-lenght.

    Parameters
    ----------
    random_seed : int
        Seed for probe positions.
    focal_length_m : float
        Focal-length of imaging reflector.
    outer_radius_m : float
        Outer radius of aperture's annulus.
    inner_radius_m : float
        Inner radius of aperture's annulus.
    scatter_center_areal_density_per_m2 : float
        Areal density of scatter-centers on imaging reflector.
    """
    assert focal_length_m > 0.0
    assert outer_radius_m > 0.0
    assert inner_radius_m > 0.0
    assert outer_radius_m > inner_radius_m
    assert scatter_center_areal_density_per_m2 > 0.0

    prng = np.random.Generator(np.random.PCG64(random_seed))

    gs = 1.0 / np.sqrt(scatter_center_areal_density_per_m2)
    sr = outer_radius_m + gs
    x_m = []
    y_m = []
    for xp_m in np.linspace(-sr, sr, (2 * sr / gs)):
        for yp_m in np.linspace(-sr, sr, (2 * sr / gs)):
            xf_m = xp_m + prng.uniform(low=-gs / 3, high=gs / 3, size=1)
            yf_m = yp_m + prng.uniform(low=-gs / 3, high=gs / 3, size=1)
            x_m.append(xf_m)
            y_m.append(yf_m)
    x_m = np.array(x_m)
    y_m = np.array(y_m)

    radius_m = np.sqrt(x_m ** 2 + y_m ** 2)

    inside_outer = radius_m <= outer_radius_m
    outside_inner = radius_m >= inner_radius_m

    in_annulus = np.logical_and(inside_outer, outside_inner)

    number_probes = in_annulus.sum()

    positions = np.zeros(shape=(number_probes, 3))
    positions[:, 0] = x_m[in_annulus]
    positions[:, 1] = y_m[in_annulus]
    positions[:, 2] = make_parabola_surface_height_m(
        distance_to_optical_axis_m=radius_m[in_annulus],
        focal_length_m=focal_length_m,
    )
    return positions


def make_mirror(
    random_seed,
    focal_length_m,
    outer_radius_m,
    inner_radius_m,
    scatter_center_areal_density_per_m2,
):
    imre = {}
    imre["random_seed"] = random_seed
    imre["focal_length_m"] = focal_length_m
    imre["outer_radius_m"] = outer_radius_m
    imre["inner_radius_m"] = inner_radius_m
    imre["diameter_m"] = 2.0 * outer_radius_m
    imre["area_m2"] = np.pi * (outer_radius_m ** 2 - inner_radius_m ** 2)
    imre[
        "scatter_center_areal_density_per_m2"
    ] = scatter_center_areal_density_per_m2
    imre["scatter_center_positions_m"] = make_probe_positions(
        random_seed=random_seed,
        focal_length_m=focal_length_m,
        outer_radius_m=outer_radius_m,
        inner_radius_m=inner_radius_m,
        scatter_center_areal_density_per_m2=scatter_center_areal_density_per_m2,
    )
    imre["num_scatter_centers"] = imre["scatter_center_positions_m"].shape[0]
    return imre


UNIT_HEX_U = np.array([1.0, 0.0, 0.0])
UNIT_HEX_V = np.array([0.5, np.sqrt(3) / 2, 0.0])


def make_feed_horn_positions(
    sensor_outer_radius_m, sensor_distance_m, feed_horn_inner_radius_m,
):
    """
    Returns the positions of feed-horns placed in a disk.

    Parameters
    ----------
    sensor_outer_radius_m : float
        Outer radius of the plane of sensors.
    sensor_distance_m : float
        This sensor's distance from the mirror's principal plane (z-axis).
    feed_horn_inner_radius_m : float
        The inner radius (hexagonal packing) of the feed-horn. This means
        the center of a neighboring feed-horn is 2 * inner radius away.
    """
    assert sensor_outer_radius_m > 0
    assert sensor_distance_m > 0
    assert feed_horn_inner_radius_m > 0

    hex_u_m = 2.0 * feed_horn_inner_radius_m * UNIT_HEX_U
    hex_v_m = 2.0 * feed_horn_inner_radius_m * UNIT_HEX_V

    feed_horn_outer_radius_m = feed_horn_inner_radius_m * (2.0 / np.sqrt(3.0))

    num = int(np.ceil(sensor_outer_radius_m / feed_horn_inner_radius_m))

    positions_m = []
    for u in np.arange(-num, num + 1):
        for v in np.arange(-num, num + 1):
            pos_xy_m = u * hex_u_m + v * hex_v_m
            radius_m = np.linalg.norm(pos_xy_m)
            if radius_m + feed_horn_outer_radius_m < sensor_outer_radius_m:
                positions_m.append(
                    pos_xy_m + np.array([0, 0, sensor_distance_m])
                )
    return np.array(positions_m)


def _area_of_hexagon(inner_radius):
    return 2.0 * np.sqrt(3.0) * inner_radius ** 2.0


def make_feed_horn_areal_density_per_m2(feed_horn_inner_radius_m):
    """
    Compute how many feed-horns can be placed in a unit of area.
    """
    feed_horn_area = _area_of_hexagon(inner_radius=feed_horn_inner_radius_m)
    return 1.0 / feed_horn_area


def make_sensor(
    sensor_outer_radius_m, sensor_distance_m, feed_horn_inner_radius_m, feed_horn_transmission,
):
    imse = {}
    imse["outer_radius_m"] = sensor_outer_radius_m
    imse["outer_diameter_m"] = 2 * sensor_outer_radius_m

    imse["feed_horn_inner_radius_m"] = feed_horn_inner_radius_m
    imse["feed_horn_positions_m"] = make_feed_horn_positions(
        sensor_outer_radius_m=sensor_outer_radius_m,
        sensor_distance_m=sensor_distance_m,
        feed_horn_inner_radius_m=imse["feed_horn_inner_radius_m"],
    )
    imse[
        "feed_horn_areal_density_per_m2"
    ] = make_feed_horn_areal_density_per_m2(
        feed_horn_inner_radius_m=imse["feed_horn_inner_radius_m"],
    )
    imse["feed_horn_transmission"] = feed_horn_transmission
    imse["num_feed_horns"] = imse["feed_horn_positions_m"].shape[0]
    imse["feed_horn_area_m2"] = 1.0 / imse["feed_horn_areal_density_per_m2"]
    return imse


def make_matrix(
    mirror, sensor, speed_of_light_m_per_s,
):
    """
    Estimate the imaging matrix which propagates spherical waves from the
    mirror's probing antennas to the sensor's feed-horns.

    Parameters
    ----------
    mirror : dict
        Positions of the huygenes probes on the imaging mirror.
    sensor : dict
        Positions of the feed horns in the image-sensor.
    speed_of_light_m_per_s : float
        The speed of light between the imaging reflector and the image-sensor.
    """
    assert speed_of_light_m_per_s > 0.0

    distances_m = scipy.spatial.distance_matrix(
        sensor["feed_horn_positions_m"], mirror["scatter_center_positions_m"],
    ).astype(np.float32)

    absolute_time_delays_s = distances_m / speed_of_light_m_per_s
    relative_time_delays_s = absolute_time_delays_s - np.min(
        absolute_time_delays_s
    )
    relative_amplitudes = (1 / distances_m ** 2) / (
        1 / distances_m ** 2
    ).mean()

    imma = {}
    imma["distances_m"] = distances_m
    imma["absolute_time_delays_s"] = absolute_time_delays_s
    imma["relative_time_delays_s"] = relative_time_delays_s
    imma["relative_amplitudes"] = relative_amplitudes
    return imma


def make_telescope(mirror, sensor, lnb, speed_of_light_m_per_s):
    tele = {}
    tele["sensor"] = sensor
    tele["mirror"] = mirror
    tele["lnb"] = lnb
    tele["matrix"] = make_matrix(
        mirror=mirror,
        sensor=sensor,
        speed_of_light_m_per_s=speed_of_light_m_per_s,
    )
    tele["trigger"] = {}
    tele["trigger"]["pixel_summation"] = find_neighbors(
        positions_xy=tele["sensor"]["feed_horn_positions_m"][:, 0:2],
        max_num_neighbors=7,
        integration_radius=tele["sensor"]["feed_horn_inner_radius_m"] * 2.1,
    )
    return tele


def propagate_electric_field_from_mirror_to_sensor(
    telescope, mirror_electric_fields, num_time_slices,
):
    mir = mirror_electric_fields

    out = {}
    out["global_start_time_s"] = mir["global_start_time_s"] + np.mean(
        telescope["matrix"]["absolute_time_delays_s"]
    )
    out["time_slice_duration_s"] = mir["time_slice_duration_s"]
    out["num_time_slices"] = num_time_slices
    out["num_antennas"] = telescope["sensor"]["num_feed_horns"]
    out["electric_fields_V_per_m"] = np.zeros(
        shape=(out["num_antennas"], out["num_time_slices"], 3),
        dtype=np.float32,
    )

    mirror_gain = (
        telescope["sensor"]["feed_horn_areal_density_per_m2"]
        / telescope["mirror"]["scatter_center_areal_density_per_m2"]
    )

    for dim in range(3):
        for ise in range(telescope["sensor"]["num_feed_horns"]):
            for imi in range(telescope["mirror"]["num_scatter_centers"]):
                time_delay = telescope["matrix"]["relative_time_delays_s"][
                    ise, imi
                ]

                slice_delay = int(
                    np.round(time_delay / out["time_slice_duration_s"])
                )

                gain = 1.0
                gain *= telescope["matrix"]["relative_amplitudes"][ise, imi]
                gain *= mirror_gain

                signal.add_first_to_second_at(
                    first=mir["electric_fields_V_per_m"][imi, :, dim] * gain,
                    second=out["electric_fields_V_per_m"][ise, :, dim],
                    at=slice_delay,
                )
    return out


def find_neighbors(positions_xy, max_num_neighbors, integration_radius):
    assert integration_radius > 0.0
    assert max_num_neighbors > 0
    tree = scipy.spatial.cKDTree(data=positions_xy)
    mask = []
    for pos_xy in positions_xy:
        dd, nn = tree.query(
            x=pos_xy,
            k=max_num_neighbors,
            distance_upper_bound=integration_radius,
        )
        nn_out = []
        for i in range(len(nn)):
            if dd[i] < integration_radius:
                nn_out.append(nn[i])
        mask.append(nn_out)
    return mask


def apply_pixel_summation(signal, pixel_summation):
    num_pixel_out = len(pixel_summation)
    out_shape = list(signal.shape)
    out_shape[0] = num_pixel_out
    out = np.zeros(shape=out_shape)
    for p in range(num_pixel_out):
        for s in pixel_summation[p]:
            out[p] += signal[s]
    return out
