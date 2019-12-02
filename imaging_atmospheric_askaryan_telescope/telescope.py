# Copyright 2017 Sebastian A. Mueller
import numpy as np
import collections
import scipy.spatial.distance
from scipy.signal import butter
from scipy.signal import lfilter
import os
from os.path import join
import json
import tarfile


def parabola_surface_height(
    distance_to_optical_axis,
    focal_length,
):
    z = 1 / (4.0 * focal_length) * distance_to_optical_axis**2
    return z


ImagingReflector = collections.namedtuple(
    'ImagingReflector', [
        'random_seed',
        'focal_length',
        'aperture_radius',
        'aperture_diameter',
        'antenna_areal_density',
        'area',
        'number_huygens_antennas',
        'huygens_antennas_positions'])


def make_ImagingReflector(
    focal_length=75,
    aperture_radius=25,
    random_seed=0,
    antenna_areal_density=3
):
    np.random.seed(random_seed)
    square_area = (2*aperture_radius)**2
    number_antennas_in_square = np.int(
        np.ceil(square_area*antenna_areal_density))
    x = np.random.uniform(
        low=-aperture_radius,
        high=+aperture_radius,
        size=number_antennas_in_square)
    y = np.random.uniform(
        low=-aperture_radius,
        high=+aperture_radius,
        size=number_antennas_in_square)
    r = np.sqrt(x**2 + y**2)
    inside_aperture = r <= aperture_radius
    number_huygens_antennas = inside_aperture.sum()
    huygens_antennas_positions = np.zeros(shape=(number_huygens_antennas, 3))
    huygens_antennas_positions[:, 0] = x[inside_aperture]
    huygens_antennas_positions[:, 1] = y[inside_aperture]
    huygens_antennas_positions[:, 2] = parabola_surface_height(
        distance_to_optical_axis=r[inside_aperture],
        focal_length=focal_length)

    return ImagingReflector(
        random_seed=random_seed,
        focal_length=focal_length,
        aperture_radius=aperture_radius,
        aperture_diameter=aperture_radius*2,
        antenna_areal_density=antenna_areal_density,
        area=np.pi*aperture_radius**2,
        number_huygens_antennas=number_huygens_antennas,
        huygens_antennas_positions=huygens_antennas_positions)


def ImagingReflector_from_dict(d):
    return make_ImagingReflector(
        focal_length=d['focal_length'],
        aperture_radius=d['aperture_radius'],
        random_seed=d['random_seed'],
        antenna_areal_density=d['antenna_areal_density'])


def ImagingReflector_to_dict(imaging_reflector):
    ir = imaging_reflector
    return {
        'focal_length': ir.focal_length,
        'aperture_radius': ir.aperture_radius,
        'random_seed': ir.random_seed,
        'antenna_areal_density': ir.antenna_areal_density}


ImageSensor = collections.namedtuple(
    'ImageSensor', [
        'focal_length_of_imaging_system',
        'image_sensor_distance',
        'pixel_inner_fov',
        'pixel_inner_radius',
        'pixel_inner_diameter',
        'pixel_outer_radius',
        'pixel_outer_diameter',
        'fov',
        'radius',
        'diameter',
        'area',
        'unit_u',
        'unit_v',
        'pixel_positions',
        'number_pixels',
        'pixel_directions',
        'antenna_areal_density'])


def make_ImageSensor(
    pixel_inner_fov=np.deg2rad(0.11),
    fov=np.deg2rad(4.4),
    focal_length_of_imaging_system=75,
    image_sensor_distance=75,
):
    pixel_inner_radius = (
        focal_length_of_imaging_system*np.tan(pixel_inner_fov/2))
    pixel_inner_diameter = 2*pixel_inner_radius

    pixel_outer_radius = pixel_inner_radius*2/np.sqrt(3)
    pixel_outer_diameter = 2*pixel_outer_radius

    radius = focal_length_of_imaging_system*np.tan(fov/2)
    area = np.pi*radius**2

    _unit_u = pixel_inner_diameter*np.array([1., 0., 0.])
    _unit_v = pixel_inner_diameter*np.array([.5, np.sqrt(3)/2, .0])
    pixels_on_diagonal = int(np.ceil(fov/pixel_inner_fov))

    pixel_positions = []
    for u in np.arange(-pixels_on_diagonal, pixels_on_diagonal + 1):
        for v in np.arange(-pixels_on_diagonal, pixels_on_diagonal + 1):
            pos_xy = u * _unit_u + v * _unit_v
            if np.linalg.norm(pos_xy) < radius:
                pixel_positions.append(
                    pos_xy + np.array([0, 0, image_sensor_distance]))

    pixel_positions = np.array(pixel_positions)
    number_pixels = pixel_positions.shape[0]
    pixel_directions = np.zeros(shape=(number_pixels, 2))
    pixel_directions = -np.arctan(
        pixel_positions[:, 0:2]/focal_length_of_imaging_system)
    antenna_areal_density = number_pixels/area

    return ImageSensor(
        focal_length_of_imaging_system=focal_length_of_imaging_system,
        image_sensor_distance=image_sensor_distance,
        pixel_inner_fov=pixel_inner_fov,
        pixel_inner_radius=pixel_inner_radius,
        pixel_inner_diameter=pixel_inner_diameter,
        pixel_outer_radius=pixel_outer_radius,
        pixel_outer_diameter=pixel_outer_diameter,
        fov=fov,
        radius=radius,
        diameter=2*radius,
        area=area,
        unit_u=_unit_u,
        unit_v=_unit_v,
        pixel_positions=pixel_positions,
        number_pixels=number_pixels,
        pixel_directions=pixel_directions,
        antenna_areal_density=antenna_areal_density)


def ImageSensor_from_dict(d):
    return make_ImageSensor(
        pixel_inner_fov=d['pixel_inner_fov'],
        fov=d['fov'],
        focal_length_of_imaging_system=d['focal_length_of_imaging_system'],
        image_sensor_distance=d['image_sensor_distance'])


def ImageSensor_to_dict(image_sensor):
    i = image_sensor
    return {
        'pixel_inner_fov': i.pixel_inner_fov,
        'fov': i.fov,
        'focal_length_of_imaging_system': i.focal_length_of_imaging_system,
        'image_sensor_distance': i.image_sensor_distance}


HuygensImagingGeometry = collections.namedtuple(
    'HuygensImagingGeometry', [
        'imaging_reflector',
        'image_sensor',
        'speed_of_light',
        'distances',
        'time_delays',
        'relative_time_delays',
        'relative_amplitudes'])


def make_HuygensImagingGeometry(
    imaging_reflector,
    image_sensor,
    speed_of_light=299792458
):
    distances = scipy.spatial.distance_matrix(
        image_sensor.pixel_positions,
        imaging_reflector.huygens_antennas_positions
    ).astype(np.float32)

    time_delays = distances/speed_of_light
    relative_time_delays = time_delays - np.min(time_delays)
    relative_amplitudes = ((1/distances**2)/(1/distances**2).mean())

    return HuygensImagingGeometry(
        imaging_reflector=imaging_reflector,
        image_sensor=image_sensor,
        speed_of_light=speed_of_light,
        distances=distances,
        time_delays=time_delays,
        relative_time_delays=relative_time_delays,
        relative_amplitudes=relative_amplitudes)


def add_first_to_second_at(first, second, at):
    if at > second.shape[0]:
        return

    end = at + first.shape[0]
    if end < 0:
        return

    if end >= second.shape[0]:
        end = second.shape[0]

    start = at
    if start < 0:
        start = 0

    second[start:end] += first[start - at:end - at]


def simulate_image_sensor_response(
    huygens_matrix,
    raw_imaging_reflector_huygens_antenna_responses,
    number_time_slices=300,
    component='north_component'
):
    raw = raw_imaging_reflector_huygens_antenna_responses
    image_sensor_response = np.zeros(
        shape=(huygens_matrix.image_sensor.number_pixels, number_time_slices)
    )

    antenna_response = raw[component]
    time_slice_duration = raw['time_slice_duration']
    antenna_start_slice_offsets = raw['antenna_start_slice_offsets']

    dish2pixel_gain = (
        huygens_matrix.image_sensor.antenna_areal_density /
        huygens_matrix.imaging_reflector.antenna_areal_density
    )

    for pix in range(huygens_matrix.image_sensor.number_pixels):
        for ref in range(
            huygens_matrix.imaging_reflector.number_huygens_antennas
        ):
            dish2pixel_time_delay = huygens_matrix.relative_time_delays[
                pix, ref
            ]
            dish2pixel_slice_delay = int(
                np.round(dish2pixel_time_delay / time_slice_duration)
            )

            A = huygens_matrix.relative_amplitudes[pix, ref]
            A *= dish2pixel_gain
            dish_record_slice_delay = antenna_start_slice_offsets[ref]

            add_first_to_second_at(
                first=antenna_response[ref, :] * A,
                second=image_sensor_response[pix, :],
                at=dish2pixel_slice_delay + dish_record_slice_delay,
            )
    return image_sensor_response


RawImageSensorResponse = collections.namedtuple(
    'RawImageSensorResponse', [
        'north',
        'west',
        'vertical',
        'number_pixels',
        'number_time_slices',
        'time_slice_duration'])


def make_RawImageSensorResponse_from_dir(
    raw_image_sensor_responses_dir,
    number_pixels,
    time_slice_duration
):
    raw_dir = raw_image_sensor_responses_dir

    with open(join(raw_dir, 'north_component.float32'), 'rb') as fin:
        north = np.fromstring(fin.read(), dtype=np.float32)
        north = north.reshape((number_pixels, north.shape[0] // number_pixels))

    with open(join(raw_dir, 'west_component.float32'), 'rb') as fin:
        west = np.fromstring(fin.read(), dtype=np.float32)
        west = west.reshape((number_pixels, west.shape[0] // number_pixels))

    with open(join(raw_dir, 'vertical_component.float32'), 'rb') as fin:
        vertical = np.fromstring(fin.read(), dtype=np.float32)
        vertical = vertical.reshape(
            (number_pixels, vertical.shape[0] // number_pixels))

    return RawImageSensorResponse(
        north=north,
        west=west,
        vertical=vertical,
        number_pixels=number_pixels,
        number_time_slices=north.shape[1],
        time_slice_duration=time_slice_duration)


Event = collections.namedtuple(
    'Event', [
        'id',
        'config',
        'time_window_probe_antenna',
        'simulation_truth',
        'imaging_reflector',
        'image_sensor',
        'raw_image_sensor_response'])


def make_Event_from_path(path):
    with open(os.path.join(path, 'config.json'), 'r') as fin:
        config = json.loads(fin.read())
    with open(os.path.join(path, 'time_window.json'), 'r') as fin:
        time_window = json.loads(fin.read())
    imaging_reflector = ImagingReflector_from_dict(
        config['imaging_reflector'])
    image_sensor = ImageSensor_from_dict(config['image_sensor'])
    id_ = config['event_id']
    simulation_truth = config['simulation_truth']
    raw_image_sensor_response = make_RawImageSensorResponse_from_dir(
        os.path.join(path, 'raw_image_sensor_response'),
        number_pixels=image_sensor.number_pixels,
        time_slice_duration=simulation_truth['time_slice_duration'])

    return Event(
        id=id_,
        config=config,
        time_window_probe_antenna=time_window,
        simulation_truth=simulation_truth,
        imaging_reflector=imaging_reflector,
        image_sensor=image_sensor,
        raw_image_sensor_response=raw_image_sensor_response)


def make_Event_from_tape_archive(path):
    with tarfile.open(path, "r") as tar_file:
        return make_next_Event_from_tape_archive(tar_file)


def make_next_Event_from_tape_archive(tar_file):
    config_tar_item = tar_file.next()
    assert config_tar_item.name == "config.json"
    config_raw = tar_file.extractfile(config_tar_item).read()
    config = json.loads(config_raw)

    imaging_reflector = ImagingReflector_from_dict(config['imaging_reflector'])
    image_sensor = ImageSensor_from_dict(config['image_sensor'])

    time_window_tar_item = tar_file.next()
    assert time_window_tar_item.name == "time_window.json"
    time_window_raw = tar_file.extractfile(time_window_tar_item).read()
    time_window = json.loads(time_window_raw)

    comps = {"north": None, "west": None, "vertical": None}
    for comp in comps:
        comp_item = tar_file.next()
        assert comp_item.name == comp + '_component.float32'
        comp_raw = tar_file.extractfile(comp_item).read()

        field = np.fromstring(comp_raw, dtype=np.float32)
        field = field.reshape((
            image_sensor.number_pixels,
            field.shape[0]//image_sensor.number_pixels))
        comps[comp] = field

    raw_image_sensor_response = RawImageSensorResponse(
        north=comps["north"],
        west=comps["west"],
        vertical=comps["vertical"],
        number_pixels=image_sensor.number_pixels,
        number_time_slices=comps["north"].shape[1],
        time_slice_duration=config['simulation_truth']['time_slice_duration'])

    return Event(
        id=config['event_id'],
        config=config,
        time_window_probe_antenna=time_window,
        simulation_truth=config['simulation_truth'],
        imaging_reflector=imaging_reflector,
        image_sensor=image_sensor,
        raw_image_sensor_response=raw_image_sensor_response)


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data, axis=axis)
    return y


def simulate_antenna_response(
    raw_image_sensor_response,
    antenna_efficiency=0.5,
    antenna_temperature=80,
    lower_frequency_cut=1.3e9,
    upper_frequency_cut=2.3e9,
    order=5,
):
    # band pass filter
    # ----------------
    risr = raw_image_sensor_response
    fs = 1.0/raw_image_sensor_response.time_slice_duration
    north = _butter_bandpass_filter(
        data=risr.north.T,
        lowcut=lower_frequency_cut,
        highcut=upper_frequency_cut,
        fs=fs,
        order=order).T
    west = _butter_bandpass_filter(
        data=risr.west.T,
        lowcut=lower_frequency_cut,
        highcut=upper_frequency_cut,
        fs=fs,
        order=order).T
    vertical = _butter_bandpass_filter(
        data=risr.vertical.T,
        lowcut=lower_frequency_cut,
        highcut=upper_frequency_cut,
        fs=fs,
        order=order).T

    # antenna efficieny
    # -----------------
    north = antenna_efficiency*north
    west = antenna_efficiency*west
    vertical = antenna_efficiency*vertical

    # antenna noise
    # -------------
    BOLTZMANN_CONSTANT = 1.38e-23
    VACUUM_IMPEDANCE = 120.0*np.pi
    antenna_bandwidth = upper_frequency_cut - lower_frequency_cut

    antenna_noise_power = antenna_temperature*(
        BOLTZMANN_CONSTANT*antenna_bandwidth)
    noise_e_field = np.sqrt(antenna_noise_power*VACUUM_IMPEDANCE)

    north += np.random.normal(
        loc=0.0,
        scale=noise_e_field,
        size=(north.shape[0], north.shape[1]))
    west += np.random.normal(
        loc=0.0,
        scale=noise_e_field,
        size=(west.shape[0], west.shape[1]))
    vertical += np.random.normal(
        loc=0.0,
        scale=noise_e_field,
        size=(vertical.shape[0], vertical.shape[1]))

    return RawImageSensorResponse(
        north=north.astype(np.float32),
        west=west.astype(np.float32),
        vertical=vertical.astype(np.float32),
        number_pixels=risr.number_pixels,
        number_time_slices=north.shape[1],
        time_slice_duration=risr.time_slice_duration)
