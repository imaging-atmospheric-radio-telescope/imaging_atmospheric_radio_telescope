# Copyright 2017 Sebastian A. Mueller
import numpy as np
import scipy
from scipy import spatial
import scipy.spatial.distance
import os
from . import signal


def parabola_surface_height(
    distance_to_optical_axis, focal_length,
):
    """
    Parameters
    ----------
    distance_to_optical_axis : float
        The distance to the parabola's optical axis for where the height of
        the parabola is estimated.
    focal_length : float
        The parabola's focal-length.
    """
    z = 1 / (4.0 * focal_length) * distance_to_optical_axis ** 2
    return z


def make_probe_positions(
    random_seed=0,
    focal_length=75,
    outer_radius=25,
    inner_radius=10,
    probe_areal_density=3,
):
    """
    Returns the randomly drawn positions of huygens probes on a parabolic
    imaging reflector. The x-, y-positions can be limited in an anulus with
    an innner, and outer radius. The z-position is computed from the
    focal-lenght.

    Parameters
    ----------
    random_seed : int
        Seed for probe positions.
    focal_length : float
        Focal-length of imaging reflector.
    outer_radius : float
        Outer radius of aperture's annulus.
    inner_radius : float
        Inner radius of aperture's annulus.
    probe_areal_density : float
        Density of probes per area in imaging reflector.
    """
    assert focal_length > 0.0
    assert outer_radius > 0.0
    assert inner_radius > 0.0
    assert outer_radius > inner_radius
    assert probe_areal_density > 0.0

    prng = np.random.Generator(np.random.PCG64(random_seed))

    gs = 1.0 / np.sqrt(probe_areal_density)
    sr = outer_radius + gs
    x = []
    y = []
    for xp in np.linspace(-sr, sr, (2 * sr / gs)):
        for yp in np.linspace(-sr, sr, (2 * sr / gs)):
            xf = xp + prng.uniform(low=-gs / 3, high=gs / 3, size=1)
            yf = yp + prng.uniform(low=-gs / 3, high=gs / 3, size=1)
            x.append(xf)
            y.append(yf)
    x = np.array(x)
    y = np.array(y)

    r = np.sqrt(x ** 2 + y ** 2)

    inside_outer = r <= outer_radius
    outside_inner = r >= inner_radius

    in_annulus = np.logical_and(inside_outer, outside_inner)

    number_probes = in_annulus.sum()

    positions = np.zeros(shape=(number_probes, 3))
    positions[:, 0] = x[in_annulus]
    positions[:, 1] = y[in_annulus]
    positions[:, 2] = parabola_surface_height(
        distance_to_optical_axis=r[in_annulus], focal_length=focal_length
    )
    return positions


def make_mirror(
    random_seed=0,
    focal_length=75,
    outer_radius=25,
    inner_radius=12,
    probe_areal_density=3,
):
    imre = {}
    imre["random_seed"] = random_seed
    imre["focal_length"] = focal_length
    imre["outer_radius"] = outer_radius
    imre["inner_radius"] = inner_radius
    imre["diameter"] = 2.0 * outer_radius
    imre["area"] = np.pi * (outer_radius ** 2 - inner_radius ** 2)
    imre["antenna_areal_density"] = probe_areal_density
    imre["antenna_positions"] = make_probe_positions(
        random_seed=random_seed,
        focal_length=focal_length,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        probe_areal_density=probe_areal_density,
    )
    imre["num_antennas"] = imre["antenna_positions"].shape[0]
    return imre


UNIT_HEX_U = np.array([1.0, 0.0, 0.0])
UNIT_HEX_V = np.array([0.5, np.sqrt(3) / 2, 0.0])


def make_feed_horn_positions(
    sensor_outer_radius, sensor_distance, feed_horn_inner_radius,
):
    """
    Returns the positions of feed-horns placed in a disk.

    Parameters
    ----------
    sensor_outer_radius : float
        Outer radius of the plane of sensors.
    sensor_distance : float
        This sensor's distance from the mirror's principal plane (z-axis).
    feed_horn_inner_radius : float
        The inner radius (hexagonal packing) of the feed-horn. This means
        the center of a neighboring feed-horn is 2 * inner radius away.
    """
    assert sensor_outer_radius > 0
    assert sensor_distance > 0
    assert feed_horn_inner_radius > 0

    hex_u = 2.0 * feed_horn_inner_radius * UNIT_HEX_U
    hex_v = 2.0 * feed_horn_inner_radius * UNIT_HEX_V

    feed_horn_outer_radius = feed_horn_inner_radius * (2.0 / np.sqrt(3.0))

    num = int(np.ceil(sensor_outer_radius / feed_horn_inner_radius))

    positions = []
    for u in np.arange(-num, num + 1):
        for v in np.arange(-num, num + 1):
            pos_xy = u * hex_u + v * hex_v
            r = np.linalg.norm(pos_xy)
            if r + feed_horn_outer_radius < sensor_outer_radius:
                positions.append(pos_xy + np.array([0, 0, sensor_distance]))
    return np.array(positions)


def _area_of_hexagon(inner_radius):
    return 2.0 * np.sqrt(3.0) * inner_radius ** 2.0


def feed_horn_areal_density(feed_horn_inner_radius):
    """
    Compute how many feed-horns can be placed in a unit of area.
    """
    feed_horn_area = _area_of_hexagon(inner_radius=feed_horn_inner_radius)
    return 1.0 / feed_horn_area


def make_sensor(
    sensor_outer_radius, sensor_distance, feed_horn_inner_radius,
):
    imse = {}
    imse["outer_radius"] = sensor_outer_radius
    imse["outer_diameter"] = 2 * sensor_outer_radius

    imse["antenna_inner_radius"] = feed_horn_inner_radius
    imse["antenna_positions"] = make_feed_horn_positions(
        sensor_outer_radius=sensor_outer_radius,
        sensor_distance=sensor_distance,
        feed_horn_inner_radius=imse["antenna_inner_radius"],
    )
    imse["antenna_areal_density"] = feed_horn_areal_density(
        feed_horn_inner_radius=imse["antenna_inner_radius"],
    )
    imse["num_antennas"] = imse["antenna_positions"].shape[0]
    imse["antenna_area"] = 1.0 / imse["antenna_areal_density"]
    imse["area"] = imse["antenna_area"] * imse["num_antennas"]
    return imse


def make_matrix(
    mirror, sensor, speed_of_light,
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
    speed_of_light : float
        The speed of light between the imaging reflector and the image-sensor.
    """
    assert speed_of_light > 0.0

    distances = scipy.spatial.distance_matrix(
        sensor["antenna_positions"], mirror["antenna_positions"],
    ).astype(np.float32)

    absolute_time_delays = distances / speed_of_light
    relative_time_delays = absolute_time_delays - np.min(absolute_time_delays)
    relative_amplitudes = (1 / distances ** 2) / (1 / distances ** 2).mean()

    imma = {}
    imma["distances"] = distances
    imma["absolute_time_delays"] = absolute_time_delays
    imma["relative_time_delays"] = relative_time_delays
    imma["relative_amplitudes"] = relative_amplitudes
    return imma


def make_telescope(mirror, sensor, lnb, speed_of_light):
    tele = {}
    tele["sensor"] = sensor
    tele["mirror"] = mirror
    tele["lnb"] = lnb
    tele["matrix"] = make_matrix(
        mirror=mirror, sensor=sensor, speed_of_light=speed_of_light,
    )
    tele["trigger"] = {}
    tele["trigger"]["pixel_summation"] = find_neighbors(
        positions_xy=tele["sensor"]["antenna_positions"][:, 0:2],
        max_num_neighbors=7,
        integration_radius=tele["sensor"]["antenna_inner_radius"] * 2.1,
    )
    return tele


def propagate_electric_field_from_mirror_to_sensor(
    telescope, mirror_electric_fields, num_time_slices,
):
    mir = mirror_electric_fields

    out = {}
    out["global_start_time"] = mir["global_start_time"] + np.mean(
        telescope["matrix"]["absolute_time_delays"]
    )
    out["time_slice_duration"] = mir["time_slice_duration"]
    out["num_time_slices"] = num_time_slices
    out["num_antennas"] = telescope["sensor"]["num_antennas"]
    out["electric_fields"] = np.zeros(
        shape=(out["num_antennas"], out["num_time_slices"], 3),
        dtype=np.float32,
    )

    mirror_gain = (
        telescope["sensor"]["antenna_areal_density"]
        / telescope["mirror"]["antenna_areal_density"]
    )

    for dim in range(3):
        for ise in range(telescope["sensor"]["num_antennas"]):
            print(dim, ise)
            for imi in range(telescope["mirror"]["num_antennas"]):
                time_delay = telescope["matrix"]["relative_time_delays"][
                    ise, imi
                ]

                slice_delay = int(
                    np.round(time_delay / out["time_slice_duration"])
                )

                gain = 1.0
                gain *= telescope["matrix"]["relative_amplitudes"][ise, imi]
                gain *= mirror_gain

                signal.add_first_to_second_at(
                    first=mir["electric_fields"][imi, :, dim] * gain,
                    second=out["electric_fields"][ise, :, dim],
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
