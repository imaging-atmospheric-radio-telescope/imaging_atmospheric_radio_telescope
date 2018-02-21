# Copyright 2017 Sebastian A. Mueller
import numpy as np
import tempfile
import os
import subprocess
import shutil
import json

from . import steering_card_utils
from . import telescope
from . import coreas_bridge


def make_coreas_antenna_list(imaging_reflector):
    template_line = "AntennaPosition = {x:2f}\t{y:2f}\t{z:2f}\t "
    template_line += "huygens_antenna_{antenna_idx:06d}\n"
    antenna_list = ''
    for i in range(imaging_reflector.huygens_antennas_positions.shape[0]):
        antenna_list += template_line.format(
            x=imaging_reflector.huygens_antennas_positions[i, 0] * 1e2,
            y=imaging_reflector.huygens_antennas_positions[i, 1] * 1e2,
            z=imaging_reflector.huygens_antennas_positions[i, 2] * 1e2,
            antenna_idx=i,
        )
    return antenna_list


def estimate_start_time_from_antnna_response(
    raw_time,
    raw_field_components
):
    num_components = raw_field_components.shape[1]
    max_position_times = np.zeros(num_components)
    for component in range(num_components):
        first_slice = np.min(np.nonzero(raw_field_components[:, component]))
        max_position_times[component] = raw_time[first_slice]
    return np.median(max_position_times)


def simulate_air_shower_and_imaging_reflector_response(
    corsika_coreas_executable_path,
    out_event_dir,
    event_id,
    primary_particle_id,
    energy,
    zenith_distance,
    azimuth,
    observation_level_altitude,
    core_position_on_observation_level_north,
    core_position_on_observation_level_west,
    time_slice_duration,
    imaging_reflector,
    time_slice_duration_of_probe=2e-9,
    time_lower_boundary_of_probe=-1000e-6,
    time_upper_boundary_of_probe=25e-6,
    time_window_duration=4e-7,
    fraction_of_time_window_to_be_warm_up_time=0.06
):
    '''
    Simulates the electric-field-strength caused by an air-shower at the
    support-poisitions of the imaging-reflector. The result and log-files of the
    air-shower-simulation are stored in out_event_dir.

    The time-window for the electric-field-strength is estimated in advance.
    First the air-shower is simulated and the response of one single
    probe-antenna is simulated. Based on the time-series of the probe-antenna,
    the time-window is estimated. Second, the air-shower is simulated again and
    but this time with all support-antennas of the imaging-reflector using the
    time-window estimated in the first step.
    '''
    probe = telescope.ImagingReflector()
    probe.focal_length = 1
    probe.number_huygens_antennas = 1
    probe.huygens_antennas_positions = np.zeros(shape=(1,3))
    probe.aperture_radius = np.sqrt(1/np.pi)
    probe.aperture_diameter = 2*probe.aperture_radius
    probe.area = 1
    probe.antenna_areal_density = 1

    os.makedirs(out_event_dir, exist_ok=True)
    probe_event_dir = os.path.join(
        out_event_dir,
        'time_window_probe_antenna')

    simulate_air_shower_and_imaging_reflector_response_manual(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_event_dir=probe_event_dir,
        event_id=event_id,
        primary_particle_id=primary_particle_id,
        energy=energy,
        zenith_distance=zenith_distance,
        azimuth=azimuth,
        observation_level_altitude=observation_level_altitude,
        core_position_on_observation_level_north=
            core_position_on_observation_level_north,
        core_position_on_observation_level_west=
            core_position_on_observation_level_west,
        time_slice_duration=time_slice_duration_of_probe,
        imaging_reflector=probe,
        coreas_time_boundaries={
            'automatic_time_boundaries': 0,
            'time_lower_boundary': time_lower_boundary_of_probe,
            'time_upper_boundary': time_upper_boundary_of_probe,
        }
    )

    probe_response = coreas_bridge.read_all_raw_time_series(
        os.path.join(probe_event_dir,
            'raw_imaging_reflector_huygens_antenna_responses'))

    start_time = estimate_start_time_from_antnna_response(
        raw_time=probe_response[0, :, 0],
        raw_field_components=probe_response[0, :, 1:4])

    f = fraction_of_time_window_to_be_warm_up_time
    time_lower_boundary = start_time - f*time_window_duration
    time_upper_boundary = start_time + (1 - f)*time_window_duration

    with open(os.path.join(out_event_dir, 'time_window.json'), 'wt') as fout:
        fout.write(json.dumps({
                "start_time": start_time,
                "time_lower_boundary": time_lower_boundary,
                "time_upper_boundary": time_upper_boundary,
            }))

    simulate_air_shower_and_imaging_reflector_response_manual(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_event_dir=out_event_dir,
        event_id=event_id,
        primary_particle_id=primary_particle_id,
        energy=energy,
        zenith_distance=zenith_distance,
        azimuth=azimuth,
        observation_level_altitude=observation_level_altitude,
        core_position_on_observation_level_north=
            core_position_on_observation_level_north,
        core_position_on_observation_level_west=
            core_position_on_observation_level_west,
        time_slice_duration=time_slice_duration,
        imaging_reflector=imaging_reflector,
        coreas_time_boundaries={
            'automatic_time_boundaries': 0,
            'time_lower_boundary': time_lower_boundary,
            'time_upper_boundary': time_upper_boundary,
        }
    )


def simulate_air_shower_and_imaging_reflector_response_manual(
    corsika_coreas_executable_path,
    out_event_dir,
    event_id,
    primary_particle_id,
    energy,
    zenith_distance,
    azimuth,
    observation_level_altitude,
    core_position_on_observation_level_north,
    core_position_on_observation_level_west,
    time_slice_duration,
    imaging_reflector,
    coreas_time_boundaries=steering_card_utils.DEFAULT_COREAS_TIME_BOUNDARIES
):
    with tempfile.TemporaryDirectory(prefix='corsika_coreas_') as tmp_dir:
        tmp_run_dir = os.path.join(tmp_dir, 'run')

        shutil.copytree(
            os.path.dirname(corsika_coreas_executable_path),
            os.path.abspath(tmp_run_dir),
            symlinks=False
        )

        # tmp paths
        tmp_corsika_coreas_executable_path = os.path.join(
            tmp_run_dir,
            os.path.basename(corsika_coreas_executable_path)
        )
        tmp_corsika_steering_card_path = os.path.join(
            tmp_run_dir, 'RUN{:06}.inp'.format(event_id)
        )
        tmp_coreas_steering_card_path = os.path.join(
            tmp_run_dir, 'SIM{:06}.reas'.format(event_id)
        )
        tmp_coreas_antenna_list_path = os.path.join(
            tmp_run_dir, 'SIM{:06}.list'.format(event_id)
        )
        tmp_coreas_raw_antenna_output_dir = os.path.join(
            tmp_run_dir, 'SIM{:06}_coreas'.format(event_id)
        )

        with open(tmp_corsika_steering_card_path, 'wt') as fout:
            fout.write(
                steering_card_utils.make_corsika_steering_card(
                    event_id=event_id,
                    primary_particle_id=primary_particle_id,
                    energy=energy,
                    zenith_distance=zenith_distance,
                    azimuth=azimuth,
                    observation_level_altitude=observation_level_altitude,
                )
            )

        with open(tmp_coreas_steering_card_path, 'wt') as fout:
            fout.write(
                steering_card_utils.make_coreas_steering_card(
                    core_position_on_observation_level_north=(
                        core_position_on_observation_level_north
                    ),
                    core_position_on_observation_level_west=(
                        core_position_on_observation_level_west
                    ),
                    observation_level_altitude=observation_level_altitude,
                    time_slice_duration=time_slice_duration,
                    time_boundaries=coreas_time_boundaries
                )
            )

        with open(tmp_coreas_antenna_list_path, 'wt') as fout:
            fout.write(
                make_coreas_antenna_list(imaging_reflector=imaging_reflector)
            )

        corsika_steering_card_pipe, pwrite = os.pipe()
        with open(tmp_corsika_steering_card_path, 'rt') as fin:
            os.write(pwrite, str.encode(fin.read()))
            os.close(pwrite)

        # output paths
        os.makedirs(out_event_dir, exist_ok=True)
        out_corsika_coreas_dir = os.path.join(out_event_dir, 'corsika_coreas/')
        os.makedirs(out_corsika_coreas_dir)

        out_corsika_o_path = os.path.join(
            out_corsika_coreas_dir, 'corsika.std_out'
        )
        out_corsika_e_path = os.path.join(
            out_corsika_coreas_dir, 'corsika.std_error'
        )
        out_antenna_output_dir = os.path.join(
            out_event_dir, 'raw_imaging_reflector_huygens_antenna_responses/'
        )

        with open(out_corsika_o_path, 'w') as corsika_o, \
                open(out_corsika_e_path, 'w') as corsika_e:
            subprocess.call(
                tmp_corsika_coreas_executable_path,
                stdin=corsika_steering_card_pipe,
                stdout=corsika_o,
                stderr=corsika_e,
                cwd=tmp_run_dir
            )

        shutil.move(
            tmp_coreas_raw_antenna_output_dir,
            out_antenna_output_dir
        )

        shutil.move(
            tmp_coreas_antenna_list_path,
            os.path.join(
                out_corsika_coreas_dir,
                os.path.basename(tmp_coreas_antenna_list_path)
            )
        )

        shutil.move(
            tmp_coreas_steering_card_path,
            os.path.join(
                out_corsika_coreas_dir,
                os.path.basename(tmp_coreas_steering_card_path)
            )
        )

        shutil.move(
            tmp_corsika_steering_card_path,
            os.path.join(
                out_corsika_coreas_dir,
                os.path.basename(tmp_corsika_steering_card_path)
            )
        )

        # input('wait to inspect the tmp directory')


class NumPyJSONEncoder(json.JSONEncoder):
    '''
    By mgilson, Software Engineer at Argo AI, 2017
    Handles numpy number types correctly
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumPyJSONEncoder, self).default(obj)


def simulate_event(
    corsika_coreas_executable_path,
    out_event_dir,
    event_id,
    primary_particle_id,
    energy,
    zenith_distance,
    azimuth,
    observation_level_altitude,
    core_position_on_observation_level_north,
    core_position_on_observation_level_west,
    time_slice_duration,
    imaging_reflector,
    image_sensor
):
    '''
    Does a full simulation of a single event from the shower to the sensor response.
    Corsika -> Coreas -> Reflector -> Sensor Response.
    Output will be written into OUT_EVENT_DIR.
    '''
    simulate_air_shower_and_imaging_reflector_response(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_event_dir=out_event_dir,
        event_id=event_id,
        primary_particle_id=primary_particle_id,
        energy=energy,
        zenith_distance=zenith_distance,
        azimuth=azimuth,
        observation_level_altitude=observation_level_altitude,
        core_position_on_observation_level_north=(
            core_position_on_observation_level_north
        ),
        core_position_on_observation_level_west=(
            core_position_on_observation_level_west
        ),
        time_slice_duration=time_slice_duration,
        imaging_reflector=imaging_reflector,
    )

    raw_imaging_reflector_huygens_antenna_responses = (
        coreas_bridge.read_electric_field_on_imaging_reflector(
            path=os.path.join(
                out_event_dir,
                'raw_imaging_reflector_huygens_antenna_responses'
            )
        )
    )

    huygens_matrix = telescope.HuygensImagingGeometry(
        imaging_reflector=imaging_reflector,
        image_sensor=image_sensor
    )

    image_sensor_responses = {}
    for component in ['north_component', 'west_component', 'vertical_component']:
        sensor_response = telescope.simulate_image_sensor_response(
            huygens_matrix=huygens_matrix,
            raw_imaging_reflector_huygens_antenna_responses=(
                raw_imaging_reflector_huygens_antenna_responses
            ),
            number_time_slices=300,
            component=component
        )
        image_sensor_responses[component] = (sensor_response)

    out_image_sensor_response_dir = os.path.join(
        out_event_dir,
        'raw_image_sensor_response'
    )
    os.makedirs(out_image_sensor_response_dir)

    for component in [
        'north_component',
        'west_component',
        'vertical_component'
    ]:
        out_component_path = os.path.join(
            out_image_sensor_response_dir,
            component + '.float32'
        )
        with open(out_component_path, 'wb') as fout:
            fout.write(np.float32(image_sensor_responses[component]).tobytes())

    config = {
        'event_id': event_id,
        'simulation_truth': {
            'primary_particle_id': primary_particle_id,
            'energy': energy,
            'zenith_distance': zenith_distance,
            'azimuth': azimuth,
            'observation_level_altitude': observation_level_altitude,
            'core_position_on_observation_level_north': (
                core_position_on_observation_level_north
            ),
            'core_position_on_observation_level_west': (
                core_position_on_observation_level_west
            ),
            'time_slice_duration': time_slice_duration,
        },
        'image_sensor': telescope.image_sensor_to_dict(image_sensor),
        'imaging_reflector': telescope.imaging_reflector_to_dict(
            imaging_reflector
        ),
    }

    out_image_sensor_config_path = os.path.join(out_event_dir, 'config.json')

    with open(out_image_sensor_config_path, 'w') as fout:
        fout.write(json.dumps(config, indent=4, cls=NumPyJSONEncoder))


def sample_zenith_distance(
    min_zenith_distance=np.deg2rad(1),
    max_zenith_distance=np.deg2rad(5),
    size=100
):
    v_min = (np.cos(min_zenith_distance) + 1) / 2
    v_max = (np.cos(max_zenith_distance) + 1) / 2
    v = np.random.uniform(low=v_min, high=v_max, size=size)
    return np.arccos(2 * v - 1)


def sample_2D_points_within_radius(radius, size):
    rho = np.sqrt(np.random.uniform(0, 1, size)) * radius
    phi = np.random.uniform(0, 2 * np.pi, size)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def event_parameter_distribution(
    number_events=960,
    primary_particle_id=1,
    energy=[100, 1000],
    azimuth=np.deg2rad([0.0, 360]),
    zenith_distance=np.deg2rad([0.0, 1.5]),
    observation_level_altitude=2200,
    core_position_on_observation_level_max_scatter_radius=100,
    time_slice_duration=2e-10,
):
    max_r = core_position_on_observation_level_max_scatter_radius

    core_north, core_west = sample_2D_points_within_radius(
        radius=core_position_on_observation_level_max_scatter_radius,
        size=number_events
    )

    return {
        'event_id': np.arange(1, number_events + 1),
        'time_slice_duration': np.repeat(time_slice_duration, number_events),
        'primary_particle_id': np.repeat(primary_particle_id, number_events),
        'energy': np.random.uniform(
            low=energy[0],
            high=energy[1],
            size=number_events
        ),
        'observation_level_altitude': np.repeat(
            observation_level_altitude,
            number_events
        ),
        'core_position_on_observation_level_north': core_north,
        'core_position_on_observation_level_west': core_west,
        'azimuth': np.random.uniform(
            low=azimuth[0],
            high=azimuth[1],
            size=number_events
        ),
        'zenith_distance': sample_zenith_distance(
            min_zenith_distance=zenith_distance[0],
            max_zenith_distance=zenith_distance[1],
            size=number_events
        ),
    }
