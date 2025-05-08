# Copyright 2017 Sebastian A. Mueller
import numpy as np
import scipy
from scipy import spatial
import scipy.spatial.distance
import os
import copy
import binning_utils
from . import utils
from . import signal
from . import time_series


def make_parabola_surface_height_m(
    distance_to_optical_axis_m,
    focal_length_m,
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
    z = 1 / (4.0 * focal_length_m) * distance_to_optical_axis_m**2
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
    imaging reflector. The x-, y-positions can be limited in an annulus with
    an inner, and outer radius. The z-position is computed from the
    focal-length.

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
    _N = int(np.ceil(2 * sr / gs))
    for xp_m in np.linspace(-sr, sr, _N):
        for yp_m in np.linspace(-sr, sr, _N):
            xf_m = xp_m + prng.uniform(low=-gs / 3, high=gs / 3, size=1)
            yf_m = yp_m + prng.uniform(low=-gs / 3, high=gs / 3, size=1)
            x_m.append(xf_m)
            y_m.append(yf_m)
    x_m = np.array(x_m)
    y_m = np.array(y_m)

    radius_m = np.sqrt(x_m**2 + y_m**2)

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
    imre["area_m2"] = np.pi * (outer_radius_m**2 - inner_radius_m**2)
    imre["scatter_center_positions_m"] = make_probe_positions(
        random_seed=random_seed,
        focal_length_m=focal_length_m,
        outer_radius_m=outer_radius_m,
        inner_radius_m=inner_radius_m,
        scatter_center_areal_density_per_m2=scatter_center_areal_density_per_m2,
    )
    imre["num_scatter_centers"] = imre["scatter_center_positions_m"].shape[0]
    imre["scatter_center_area_m2"] = (
        imre["area_m2"] / imre["num_scatter_centers"]
    )
    return imre


UNIT_HEX_U = np.array([1.0, 0.0, 0.0])
UNIT_HEX_V = np.array([0.5, np.sqrt(3) / 2, 0.0])


def make_feed_horn_positions(
    sensor_outer_radius_m,
    sensor_distance_m,
    feed_horn_inner_radius_m,
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


def regular_polygon(n, rotation_rad=0):
    x = []
    y = []
    for phi in np.linspace(0, 2.0 * np.pi, n, endpoint=False):
        x.append(np.cos(phi + rotation_rad))
        y.append(np.sin(phi + rotation_rad))
    return np.array([x, y, np.zeros(n)]).T


def make_feed_horn_sub_scatter(num, inner_radius_m):
    assert inner_radius_m > 0.0
    assert num > 0

    if num == 1:
        return np.array([[0, 0, 0]])
    elif num == 2:
        assert False, "Not a good idea."
    elif num == 3:
        return regular_polygon(num) * inner_radius_m / 2
    elif num == 4:
        return (
            regular_polygon(num, rotation_rad=np.deg2rad(30))
            * inner_radius_m
            / 2
        )
    elif num == 5:
        xy = regular_polygon(4)
        return np.vstack([xy, [0, 0, 0]]) * inner_radius_m * (3 / 5)
    elif num == 6:
        return (
            regular_polygon(num, rotation_rad=np.deg2rad(30))
            * inner_radius_m
            / 2
        )
    elif num == 7:
        xy = regular_polygon(6, rotation_rad=np.deg2rad(30))
        return np.vstack([xy, [0, 0, 0]]) * inner_radius_m * (3 / 5)
    elif num > 7:
        _A = np.pi * inner_radius_m**2
        _a = _A / num
        _r = np.sqrt(_a / np.pi)
        return make_feed_horn_positions(
            sensor_outer_radius_m=inner_radius_m,
            sensor_distance_m=0,
            feed_horn_inner_radius_m=_r,
        )


def make_sensor(
    sensor_outer_radius_m,
    sensor_distance_m,
    feed_horn_inner_radius_m,
    feed_horn_transmission,
    feed_horn_oversampling=7,
):
    imse = {}
    imse["__type__"] = "camera"
    imse["camera"] = {}

    imse["camera"]["outer_radius_m"] = sensor_outer_radius_m
    imse["camera"]["outer_diameter_m"] = 2 * sensor_outer_radius_m
    imse["camera"]["feed_horn_inner_radius_m"] = feed_horn_inner_radius_m

    imse["feed_horn_positions_m"] = make_feed_horn_positions(
        sensor_outer_radius_m=sensor_outer_radius_m,
        sensor_distance_m=sensor_distance_m,
        feed_horn_inner_radius_m=imse["camera"]["feed_horn_inner_radius_m"],
    )
    imse["relative_feed_horn_sample_positions_m"] = make_feed_horn_sub_scatter(
        num=feed_horn_oversampling,
        inner_radius_m=feed_horn_inner_radius_m,
    )

    imse["feed_horn_transmission"] = feed_horn_transmission
    imse["num_feed_horns"] = imse["feed_horn_positions_m"].shape[0]
    imse["feed_horn_area_m2"] = utils.area_of_hexagon(
        inner_radius=imse["camera"]["feed_horn_inner_radius_m"]
    )

    imse["sensor_distance_m"] = sensor_distance_m
    return imse


def _assert_almost_linear_spacing(x, relative_epsilon=1e-6):
    dx = np.gradient(x)
    assert np.std(dx) < relative_epsilon * np.mean(dx)


def make_sensor_in_region_of_interest(
    x_bin_edges_m,
    y_bin_edges_m,
    sensor_distance_m,
    feed_horn_transmission,
):
    _assert_almost_linear_spacing(x_bin_edges_m)
    _assert_almost_linear_spacing(y_bin_edges_m)

    x_bin_width_m = np.mean(np.gradient(x_bin_edges_m))
    y_bin_width_m = np.mean(np.gradient(y_bin_edges_m))
    bin_area_m2 = x_bin_width_m * y_bin_width_m

    xbin = binning_utils.Binning(x_bin_edges_m)
    ybin = binning_utils.Binning(y_bin_edges_m)

    imse = {}
    imse["__type__"] = "region_of_interest"
    imse["region_of_interest"] = {}
    imse["region_of_interest"]["x_bin_edges_m"] = np.array(x_bin_edges_m)
    imse["region_of_interest"]["y_bin_edges_m"] = np.array(y_bin_edges_m)

    imse["feed_horn_positions_m"] = []
    for ix in range(xbin["num"]):
        for iy in range(ybin["num"]):
            _x = xbin["centers"][ix]
            _y = ybin["centers"][iy]
            _z = sensor_distance_m
            imse["feed_horn_positions_m"].append([_x, _y, _z])
    imse["feed_horn_positions_m"] = np.asarray(imse["feed_horn_positions_m"])
    imse["relative_feed_horn_sample_positions_m"] = np.array([[0, 0, 0]])

    imse["num_feed_horns"] = imse["feed_horn_positions_m"].shape[0]
    imse["feed_horn_transmission"] = feed_horn_transmission
    imse["feed_horn_area_m2"] = bin_area_m2
    imse["sensor_distance_m"] = sensor_distance_m
    return imse


def make_matrix(
    mirror,
    sensor,
    speed_of_light_m_per_s,
):
    """
    Estimate the imaging matrix which propagates spherical waves from the
    mirror's probing antennas to the sensor's feed-horns.

    Parameters
    ----------
    mirror : dict
        Positions of the Huygens probes on the imaging mirror.
    sensor : dict
        Positions of the feed horns in the image-sensor.
    speed_of_light_m_per_s : float
        The speed of light between the imaging reflector and the image-sensor.
    """
    assert speed_of_light_m_per_s > 0.0

    distances_m = scipy.spatial.distance_matrix(
        sensor["feed_horn_positions_m"],
        mirror["scatter_center_positions_m"],
    ).astype(np.float32)

    imma = {}
    imma["speed_of_light_m_per_s"] = speed_of_light_m_per_s
    imma["distances_m"] = distances_m
    imma["time_delays_s"] = distances_m / speed_of_light_m_per_s
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
    if tele["sensor"]["__type__"] == "camera":
        tele["trigger"] = {}
        tele["trigger"]["pixel_summation"] = find_neighbors(
            positions_xy=tele["sensor"]["feed_horn_positions_m"][:, 0:2],
            max_num_neighbors=7,
            integration_radius=tele["sensor"]["camera"][
                "feed_horn_inner_radius_m"
            ]
            * 2.1,
        )
    return tele


def make_telescope_like_other_but_different_sensor(telescope, sensor):
    return make_telescope(
        mirror=copy.copy(telescope["mirror"]),
        sensor=sensor,
        lnb=copy.copy(telescope["lnb"]),
        speed_of_light_m_per_s=copy.copy(
            telescope["matrix"]["speed_of_light_m_per_s"]
        ),
    )


def propagate_electric_field_from_mirror_to_sensor(
    telescope,
    mirror_electric_fields,
    num_time_slices,
):
    """
    # How to add up the electric fields ?
    #====================================
    Delays of the fields are defined by the distances between the
    scatter-centers on the mirror and the feed-horns in the sensor.

    The electric field amplitude weights are:
        sqrt(1/num_scatter_centers_in_mirror) *
        sqrt(area_mirror_scatter/area_feed_horn)
    """
    E_mirror = mirror_electric_fields
    E_sensor = time_series.zeros(
        time_slice_duration_s=E_mirror.time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=telescope["sensor"]["num_feed_horns"],
        num_components=E_mirror.num_components,
        global_start_time_s=E_mirror.global_start_time_s
        + np.min(telescope["matrix"]["time_delays_s"]),
        si_unit=E_mirror.si_unit,
        dtype=E_mirror.dtype,
    )

    relative_time_delays_s = telescope["matrix"]["time_delays_s"] - np.min(
        telescope["matrix"]["time_delays_s"]
    )

    feed_horn_area_m2 = telescope["sensor"]["feed_horn_area_m2"]
    mirror_scatter_area_m2 = telescope["mirror"]["scatter_center_area_m2"]

    N_scatter = telescope["mirror"]["num_scatter_centers"]

    e_field_scaling = np.sqrt(1.0 / N_scatter) * np.sqrt(
        mirror_scatter_area_m2 / feed_horn_area_m2
    )

    for ise in range(telescope["sensor"]["num_feed_horns"]):
        for imi in range(telescope["mirror"]["num_scatter_centers"]):
            # timing
            # ------
            time_delay_s = relative_time_delays_s[ise, imi]

            slice_delay = int(
                np.round(time_delay_s / E_sensor.time_slice_duration_s)
            )

            # amplitude
            # ---------
            for dim in range(3):
                signal.add_first_to_second_at(
                    first=E_mirror[imi, :, dim],
                    second=E_sensor[ise, :, dim],
                    at=slice_delay,
                )

        for dim in range(3):
            E_sensor[ise, :, dim] *= e_field_scaling

    return E_sensor


def propagate_electric_field_from_mirror_to_sensor2(
    telescope,
    mirror_electric_fields,
    num_time_slices,
):
    """
    ===
    """
    min_time_delay_s = np.min(telescope["matrix"]["time_delays_s"])

    E_mirror = mirror_electric_fields
    E_sensor = time_series.zeros(
        time_slice_duration_s=E_mirror.time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=telescope["sensor"]["num_feed_horns"],
        num_components=E_mirror.num_components,
        global_start_time_s=E_mirror.global_start_time_s + min_time_delay_s,
        si_unit=E_mirror.si_unit,
        dtype=E_mirror.dtype,
    )

    feed_horn_area_m2 = telescope["sensor"]["feed_horn_area_m2"]
    mirror_scatter_area_m2 = telescope["mirror"]["scatter_center_area_m2"]

    N_scatter = telescope["mirror"]["num_scatter_centers"]

    e_field_scaling = np.sqrt(1.0 / N_scatter) * np.sqrt(
        mirror_scatter_area_m2 / feed_horn_area_m2
    )

    num_sub = telescope["sensor"][
        "relative_feed_horn_sample_positions_m"
    ].shape[0]
    e_field_sub_scaling = 1.0 / num_sub

    progress = utils.PrintProgress(telescope["sensor"]["num_feed_horns"])

    for ise in range(telescope["sensor"]["num_feed_horns"]):
        progress.bump()
        for imi in range(telescope["mirror"]["num_scatter_centers"]):
            for isu in range(num_sub):
                # timing
                # ------
                sensor_vector_m = (
                    telescope["sensor"]["feed_horn_positions_m"][ise]
                    + telescope["sensor"][
                        "relative_feed_horn_sample_positions_m"
                    ][isu]
                )
                mirror_to_sensor_vector_m = (
                    sensor_vector_m
                    - telescope["mirror"]["scatter_center_positions_m"][imi]
                )
                distance_m = np.linalg.norm(mirror_to_sensor_vector_m)
                time_delay_s = (
                    distance_m / telescope["matrix"]["speed_of_light_m_per_s"]
                )
                relative_time_delay_s = time_delay_s - min_time_delay_s

                slice_delay = int(
                    np.round(
                        relative_time_delay_s / E_sensor.time_slice_duration_s
                    )
                )

                # amplitude
                # ---------
                for dim in range(3):
                    signal.add_first_to_second_at(
                        first=e_field_sub_scaling * E_mirror[imi, :, dim],
                        second=E_sensor[ise, :, dim],
                        at=slice_delay,
                    )

        for dim in range(3):
            E_sensor[ise, :, dim] *= e_field_scaling

    return E_sensor


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
