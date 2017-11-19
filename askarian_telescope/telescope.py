import numpy as np
import scipy.spatial.distance
import os
import json

def parabol_reflector_surface_height(
    distance_to_optical_axis,
    focal_length,
):  
    z = 1/(4.0*focal_length) * distance_to_optical_axis**2
    return z


class ImagingReflector():
    def __init__(
        self, 
        focal_length=75,
        aperture_radius=25,
        random_seed=0,
        antenna_areal_density=3,
    ):
        self.focal_length = focal_length
        self.aperture_radius = aperture_radius
        self.aperture_diameter = 2*self.aperture_radius
        self._random_seed = random_seed
        self.antenna_areal_density = antenna_areal_density

        self.area = self.aperture_radius**2 * np.pi
        self._init_huygens_antennas()


    def _init_huygens_antennas(self):
        square_area = (2*self.aperture_radius)**2
        number_antennas_in_square = np.int(
            np.ceil(square_area*self.antenna_areal_density)
        )

        np.random.seed(self._random_seed)

        x = np.random.uniform(
            low=-self.aperture_radius,
            high=+self.aperture_radius,
            size=number_antennas_in_square
        )

        y = np.random.uniform(
            low=-self.aperture_radius,
            high=+self.aperture_radius,
            size=number_antennas_in_square
        )

        r = np.sqrt(x**2 + y**2)
        inside_aperture = r <= self.aperture_radius

        self.number_huygens_antennas = inside_aperture.sum()

        self.huygens_antennas_positions = np.zeros(
            shape=(self.number_huygens_antennas, 3)
        )
        self.huygens_antennas_positions[:,0] = x[inside_aperture]
        self.huygens_antennas_positions[:,1] = y[inside_aperture]
        self.huygens_antennas_positions[:,2] = parabol_reflector_surface_height(
            distance_to_optical_axis=r[inside_aperture],
            focal_length=self.focal_length,
        )


    def __repr__(self):
        out = '{}('.format(self.__class__.__name__)
        out += str(self.focal_length) + 'm focal length, '
        out += str(self.aperture_diameter) + 'm aperture diameter'
        out += ')\n'
        return out


def imaging_reflector_from_dict(d):
    return ImagingReflector(
        focal_length=d['focal_length'],
        aperture_radius=d['aperture_radius'],
        random_seed=d['random_seed'],
        antenna_areal_density=d['antenna_areal_density'],
    )


def imaging_reflector_to_dict(imaging_reflector):
    ir = imaging_reflector
    return {
        'focal_length': ir.focal_length,
        'aperture_radius': ir.aperture_radius,
        'random_seed': ir._random_seed,
        'antenna_areal_density': ir.antenna_areal_density,
    }


class ImageSensor():
    def __init__(
        self,
        pixel_inner_fov=np.deg2rad(0.11),
        fov=np.deg2rad(4.4),
        focal_length_of_imaging_system=75,
        image_sensor_distance=75,
    ):
        self.focal_length_of_imaging_system = focal_length_of_imaging_system
        self.image_sensor_distance = image_sensor_distance
        
        self.pixel_inner_fov = pixel_inner_fov
        self.pixel_inner_radius = (
            self.focal_length_of_imaging_system*np.tan(self.pixel_inner_fov/2)
        )
        self.pixel_inner_diameter = 2*self.pixel_inner_radius

        self.pixel_outer_radius = self.pixel_inner_radius * 2/np.sqrt(3)
        self.pixel_outer_diameter = 2*self.pixel_outer_radius

        self.fov = fov
        self.radius = self.focal_length_of_imaging_system*np.tan(self.fov/2)
        self.diameter = 2*self.radius
        self.area = self.radius**2 * np.pi
        
        self._init_pixel_positions()
        self.antenna_areal_density = self.number_pixels/self.area


    def _init_pixel_positions(self):
        self._unit_u = self.pixel_inner_diameter*np.array([1.0, 0.0, 0.0])
        self._unit_v = self.pixel_inner_diameter*np.array([0.5, np.sqrt(3)/2, 0.0])

        pixels_on_diagonal = int(np.ceil(self.fov/self.pixel_inner_fov))

        pixel_positions = []
        for u in np.arange(-pixels_on_diagonal, pixels_on_diagonal+1):
            for v in np.arange(-pixels_on_diagonal, pixels_on_diagonal+1):
                pos_xy = u*self._unit_u + v*self._unit_v
                if np.linalg.norm(pos_xy) < self.radius:
                    pixel_positions.append(
                        pos_xy + np.array([0, 0, self.image_sensor_distance])
                    )
        self.pixel_positions = np.array(pixel_positions)
        self.number_pixels = self.pixel_positions.shape[0]
        self.pixel_directions = np.zeros(
            shape=(self.number_pixels, 2)
        )
        self.pixel_directions = np.arctan(
            self.pixel_positions[:,0:2]/self.focal_length_of_imaging_system
        )

    def __repr__(self):
        out = '{}('.format(self.__class__.__name__)
        out += str(self.number_pixels) + ' pixels, '
        out += str(np.rad2deg(self.fov)) + 'deg fov'
        out += ')\n'
        return out


def image_sensor_from_dict(d):
    return ImageSensor(
        pixel_inner_fov=d['pixel_inner_fov'],
        fov=d['fov'],
        focal_length_of_imaging_system=d['focal_length_of_imaging_system'],
        image_sensor_distance=d['image_sensor_distance']
    )


def image_sensor_to_dict(image_sensor):
    ims = image_sensor
    return {
        'pixel_inner_fov': ims.pixel_inner_fov,
        'fov': ims.fov,
        'focal_length_of_imaging_system': ims.focal_length_of_imaging_system,
        'image_sensor_distance': ims.image_sensor_distance,
    }


class HuygensImagingGeometry():
    def __init__(self, imaging_reflector, image_sensor, speed_of_light=299792458):
        self.imaging_reflector = imaging_reflector
        self.image_sensor = image_sensor
        self._speed_of_light = speed_of_light

        self.distances = scipy.spatial.distance_matrix(
            image_sensor.pixel_positions, 
            imaging_reflector.huygens_antennas_positions
        ).astype(np.float32)

        self.time_delays = self.distances/self._speed_of_light
        self.relative_time_delays = self.time_delays - self.time_delays.min()
        self.relative_amplitudes = (1/self.distances**2)/(1/self.distances**2).mean()

    def __repr__(self):
        out = '{}('.format(self.__class__.__name__)
        out += str(self._image_sensor.number_pixels) + ' pixels, '
        out += str(self._imaging_reflector.number_huygens_antennas) + ' huygenes antennas on reflector'
        out += ')\n'
        return out


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

    second[start:end] += first[start-at : end-at]


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
        huygens_matrix.image_sensor.antenna_areal_density/
        huygens_matrix.imaging_reflector.antenna_areal_density
    )

    for pix in range(huygens_matrix.image_sensor.number_pixels):
        for ref in range(huygens_matrix.imaging_reflector.number_huygens_antennas):
            
            dish2pixel_time_delay = huygens_matrix.relative_time_delays[pix, ref]
            dish2pixel_slice_delay = int(np.round(dish2pixel_time_delay/time_slice_duration))

            A = huygens_matrix.relative_amplitudes[pix, ref]
            A *= dish2pixel_gain 
            dish_record_slice_delay = antenna_start_slice_offsets[ref]
            
            add_first_to_second_at(
                first=antenna_response[ref, :]*A,
                second=image_sensor_response[pix, :],
                at=dish2pixel_slice_delay + dish_record_slice_delay,
            )
    return image_sensor_response


class RawImageSensorResponse():
    def __init__(self, raw_image_sensor_responses_dir, number_pixels, time_slice_duration):
        raw_dir = raw_image_sensor_responses_dir

        with open(os.path.join(raw_dir, 'north_component.float32'), 'rb') as fin:
            self.north = np.fromstring(fin.read(), dtype=np.float32)
            self.north = self.north.reshape(
                (number_pixels, self.north.shape[0]//number_pixels)
            )

        with open(os.path.join(raw_dir, 'west_component.float32'), 'rb') as fin:
            self.west = np.fromstring(fin.read(), dtype=np.float32)
            self.west = self.west.reshape(
                (number_pixels, self.west.shape[0]//number_pixels)
            )

        with open(os.path.join(raw_dir, 'vertical_component.float32'), 'rb') as fin:
            self.vertical = np.fromstring(fin.read(), dtype=np.float32)
            self.vertical = self.vertical.reshape(
                (number_pixels, self.vertical.shape[0]//number_pixels)
            )

        self.number_time_slices = self.north.shape[1]
        self.number_pixels = number_pixels
        self.time_slice_duration = time_slice_duration

    def __repr__(self):
        out = '{}('.format(self.__class__.__name__)
        out += str(self.number_pixels) + ' pixels, '
        out += str(self.number_time_slices) + ' time slices, '
        out += str(self.time_slice_duration*1e9) + 'ns each'
        out += ')\n'
        return out



class Event():
    def __init__(self, path):

        with open(os.path.join(path, 'config.json'), 'r') as fin:
            self._config = json.loads(fin.read())

        self.imaging_reflector = imaging_reflector_from_dict(
            self._config['imaging_reflector']
        )

        self.image_sensor = image_sensor_from_dict(
            self._config['image_sensor']
        )

        self.id = self._config['event_id']

        self.simulation_truth = self._config['simulation_truth']

        self.raw_image_sensor_response = RawImageSensorResponse(
            os.path.join(path, 'raw_image_sensor_response'),
            number_pixels=self.image_sensor.number_pixels,
            time_slice_duration=self.simulation_truth['time_slice_duration']
        )

    def __repr__(self):
        out = '{}('.format(self.__class__.__name__)
        out +=  'ID '+str(self.id)
        out += ')\n'
        return out
