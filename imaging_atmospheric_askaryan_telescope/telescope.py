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
from . import camera
from . import electric_fields
from . import theory
from . import lownoiseblock


def make_mirror_scatter_center_positions(
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
    positions[:, 2] = utils.make_parabola_surface_height_m(
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
    imre["scatter_center_positions_m"] = make_mirror_scatter_center_positions(
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


def make_sensor(
    sensor_outer_radius_m,
    sensor_distance_m,
    feed_horn_inner_radius_m,
    feed_horn_transmission,
    feed_horn_oversampling_order,
):
    return camera.make_camera(
        sensor_outer_radius_m=sensor_outer_radius_m,
        sensor_distance_m=sensor_distance_m,
        feed_horn_inner_radius_m=feed_horn_inner_radius_m,
        feed_horn_transmission=feed_horn_transmission,
        feed_horn_oversampling_order=feed_horn_oversampling_order,
    )


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

    imse["num_feed_horns"] = imse["feed_horn_positions_m"].shape[0]
    imse["feed_horn_transmission"] = feed_horn_transmission
    imse["feed_horn_area_m2"] = bin_area_m2

    imse["num_scatter_centers_per_feed_horn"] = 1
    imse["feed_horn_relative_scatter_center_positions_m"] = np.array(
        [[0, 0, 0]]
    )
    imse["lnb_relative_scatter_center_positions_m"] = np.array([[0, 0, 0]])
    imse["feed_horn_scatter_center_area_m2"] = imse["feed_horn_area_m2"]
    imse["low_noise_block_effective_area_m2"] = imse["feed_horn_area_m2"]

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

    tele["airy_half_angle_rad"] = calculate_airy_angle(tele)
    tele["airy_radius_in_focal_plane_m"] = (
        calculate_airy_disk_radius_in_focal_plane(tele)
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


def propagate_electric_field_from_mirror_to_region_of_interest_sensor(
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
                signal.add_first_to_second_at_int(
                    first=E_mirror[imi, :, dim],
                    second=E_sensor[ise, :, dim],
                    at=slice_delay,
                )

        for dim in range(3):
            E_sensor[ise, :, dim] *= e_field_scaling

    return E_sensor


def propagate_electric_field_from_mirror_to_sensor(
    telescope,
    mirror_electric_fields,
    num_time_slices,
):
    """
    ===
    """
    assert (
        0.0
        < telescope["calibration"][
            "point_spread_function_quantile_contained_in_feed_horn"
        ]["watershed"]
        <= 1.0
    )

    camera = telescope["sensor"]
    mirror = telescope["mirror"]

    min_time_delay_s = np.min(telescope["matrix"]["time_delays_s"])

    E_mirror = mirror_electric_fields
    E_feed_horns = time_series.zeros(
        time_slice_duration_s=E_mirror.time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=camera["num_feed_horns"],
        num_components=E_mirror.num_components,
        global_start_time_s=E_mirror.global_start_time_s + min_time_delay_s,
        si_unit=E_mirror.si_unit,
        dtype=E_mirror.dtype,
    )

    point_spread_function_quantile_contained_in_feed_horn_scatter = (
        telescope["calibration"][
            "point_spread_function_quantile_contained_in_feed_horn"
        ]["watershed"]
        / camera["num_scatter_centers_per_feed_horn"]
    )

    psf_containment_mirror_to_feed_horn_scatter_E_field_scaling = np.sqrt(
        point_spread_function_quantile_contained_in_feed_horn_scatter
    )

    geometric_mirror_to_feed_horn_scatter_E_field_scaling = np.sqrt(
        mirror["scatter_center_area_m2"]
        / camera["feed_horn_scatter_center_area_m2"]
    )

    summation_E_field_scaling = np.sqrt(1.0 / (mirror["num_scatter_centers"]))

    mirror_to_feed_horn_E_field_scaling = (
        geometric_mirror_to_feed_horn_scatter_E_field_scaling
        * psf_containment_mirror_to_feed_horn_scatter_E_field_scaling
        * summation_E_field_scaling
    )

    E_feed_horns_scatters = time_series.zeros(
        time_slice_duration_s=E_mirror.time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=(
            camera["num_feed_horns"]
            * camera["num_scatter_centers_per_feed_horn"]
        ),
        num_components=E_mirror.num_components,
        global_start_time_s=E_mirror.global_start_time_s + min_time_delay_s,
        si_unit=E_mirror.si_unit,
        dtype=E_mirror.dtype,
    )

    for ifh in range(camera["num_feed_horns"]):
        E_feed_horn = time_series.zeros(
            time_slice_duration_s=E_mirror.time_slice_duration_s,
            num_time_slices=num_time_slices,
            num_channels=camera["num_scatter_centers_per_feed_horn"],
            num_components=E_mirror.num_components,
            global_start_time_s=E_mirror.global_start_time_s
            + min_time_delay_s,
            si_unit=E_mirror.si_unit,
            dtype=E_mirror.dtype,
        )

        # from mirror to feed horn
        # ------------------------
        for isu in range(camera["num_scatter_centers_per_feed_horn"]):
            for imi in range(mirror["num_scatter_centers"]):

                # timing
                # ------
                feed_horn_scatter_center_position_vector_m = (
                    camera["feed_horn_positions_m"][ifh]
                    + camera["feed_horn_relative_scatter_center_positions_m"][
                        isu
                    ]
                )
                mirror_to_feed_horn_vector_m = (
                    feed_horn_scatter_center_position_vector_m
                    - mirror["scatter_center_positions_m"][imi]
                )
                mirror_sensor_distance_m = np.linalg.norm(
                    mirror_to_feed_horn_vector_m
                )

                time_delay_s = (
                    mirror_sensor_distance_m
                    / telescope["matrix"]["speed_of_light_m_per_s"]
                )
                relative_time_delay_s = time_delay_s - min_time_delay_s

                slice_delay = (
                    relative_time_delay_s / E_mirror.time_slice_duration_s
                )

                # amplitude
                # ---------
                signal.add_first_to_second_at_float(
                    first=mirror_to_feed_horn_E_field_scaling * E_mirror[imi],
                    second=E_feed_horn[isu],
                    at=slice_delay,
                )

        # copy for debug output
        # ---------------------
        for isu in range(camera["num_scatter_centers_per_feed_horn"]):
            iii = ifh * camera["num_scatter_centers_per_feed_horn"] + isu
            E_feed_horns_scatters[iii] = E_feed_horn[isu]

        # Suming up feed horn's scatter centers
        # -------------------------------------
        for isl in range(camera["num_scatter_centers_per_feed_horn"]):
            signal.add_first_to_second_at_int(
                first=element_wise_power(
                    x=E_feed_horn[isl],
                    p=2,
                ),
                second=E_feed_horns[ifh],
                at=0,
            )
        E_feed_horns[ifh] = element_wise_power(
            x=E_feed_horns[ifh],
            p=0.5,
        )

    return E_feed_horns, E_feed_horns_scatters


def element_wise_power(x, p):
    n = np.linalg.norm(x, axis=1)
    assert x.shape[0] == n.shape[0]
    out = np.zeros_like(x)
    for s in range(x.shape[0]):
        if n[s] > 0.0:
            out[s] = x[s] * n[s] ** (p - 1)
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


def calculate_airy_angle(telescope):
    f_Hz = np.mean(
        lownoiseblock.input_frequency_start_stop_Hz(telescope["lnb"])
    )

    return theory.airy_angle(
        mirror_diameter=2.0 * telescope["mirror"]["outer_radius_m"],
        wavelength=signal.frequency_to_wavelength(f_Hz),
    )


def calculate_airy_disk_radius_in_focal_plane(telescope):
    theta_rad = calculate_airy_angle(telescope=telescope)
    return np.arctan(theta_rad) * telescope["mirror"]["focal_length_m"]
