import numpy as np
from collections import OrderedDict
import tempfile
import os
import subprocess
import shutil
import json

from . import telescope
from . import coreas_bridge

"""
In CORSIKA COREAS, there must only be one shower per CORSIKA run.
"""


def make_corsika_steering_card(
    event_id=1,
    primary_particle_id=14,
    energy=100,
    zenith_distance=0.0,
    azimuth=0.0,
    observation_level_altitude=2200,
):
    zd_deg = np.rad2deg(zenith_distance)
    az_deg = np.rad2deg(azimuth)
    obs_level_cm = observation_level_altitude*1e2

    sc = "RUNNR {:d}\n".format(event_id)
    sc += "EVTNR {:d}\n".format(1)
    sc += "SEED {:d} 0 0\n".format(event_id+0)
    sc += "SEED {:d} 0 0\n".format(event_id+1)
    sc += "SEED {:d} 0 0\n".format(event_id+2)
    sc += "PRMPAR {:d}\n".format(primary_particle_id)

    sc += "ERANGE {0:.3E} {0:.3E}\n".format(energy, energy)
    sc += "ESLOPE 0\n"
    sc += "THETAP {0:.3E} {0:.3E}\n".format(zd_deg, zd_deg)
    sc += "PHIP {0:.3E} {0:.3E}\n".format(az_deg, az_deg)

    sc += "ECUTS 3.000E-01 3.000E-01 4.010E-04 4.010E-04\n"
    sc += "ELMFLG T T\n"
    sc += "NSHOW 1\n"
    sc += "DIRECT './'\n"
    sc += "OBSLEV {0:.3E}\n".format(obs_level_cm)

    sc += "ECTMAP 1.000E+05\n"
    sc += "STEPFC 1.000E+00\n"
    sc += "MUMULT T\n"
    sc += "MUADDI T\n"
    sc += "PAROUT F  F\n"
    sc += "MAXPRT 1\n"
    sc += "MAGNET 19.71 -14.18\n"
    sc += "LONGI T   5.  T  T\n"
    sc += "RADNKG 5.000E+05\n"
    sc += "DATBAS F\n"

    sc += "EXIT\n"
    return sc


def make_coreas_steering_card(
    core_position_on_observation_level_north=0.0,
    core_position_on_observation_level_west=0.0,
    observation_level_altitude=2200,
    time_slice_duration=2e-10,
):
    core_north_cm = core_position_on_observation_level_north*1e2
    core_west_cm = core_position_on_observation_level_west*1e2
    obs_level_cm = observation_level_altitude*1e2

    sc = "# CoREAS V1.1 by Tim Huege <tim.huege@kit.edu> with contributions "
    sc += "by Marianne Ludwig and Clancy James - parameter file\n"
    sc += "\n"
    sc += "# parameters setting up the spatial observer configuration:\n"
    sc += "\n"
    sc += "CoreCoordinateNorth = {0:.6E} ; in cm\n".format(core_north_cm)
    sc += "CoreCoordinateWest = {0:.6E} ; in cm\n".format(core_west_cm)
    sc += "CoreCoordinateVertical = {0:.6E} ; in cm\n".format(obs_level_cm)
    sc += "\n"
    sc += "# parameters setting up the temporal observer configuration:\n"
    sc += "\n"
    sc += "TimeResolution = {0:.6E} ; in s\n".format(time_slice_duration)
    sc += "AutomaticTimeBoundaries = 4e-07 ; "
    sc += "0: off, x: automatic boundaries with width x in s\n"
    sc += "TimeLowerBoundary = -1 ; "
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "TimeUpperBoundary = 1 ; "
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "ResolutionReductionScale = 0 ; "
    sc += "0: off, x: decrease time resolution linearly every x cm in radius\n"
    sc += "\n"
    sc += "# parameters setting up the simulation functionality:\n"
    sc += "GroundLevelRefractiveIndex = 1.000292 ; "
    sc += "specify refractive index at 0 m asl\n"
    sc += "\n"
    sc += "# event information for Offline simulations:\n"
    sc += "\n"
    sc += "EventNumber = -1\n"
    sc += "RunNumber = -1\n"
    sc += "GPSSecs = 0\n"
    sc += "CoreEastingOffline = 0 ; in meters\n"
    sc += "CoreNorthingOffline = 0 ; in meters\n"
    sc += "CoreVerticalOffline = 0 ; in meters\n"
    sc += "RotationAngleForMagfieldDeclination = 0     ; in degrees\n"
    sc += "Comment = \n"
    sc += "\n"
    sc += "# event information for your convenience and backwards "
    sc += "compatibility with other software, these values are not used as "
    sc += "input parameters for the simulation:\n"
    sc += "\n"
    return sc


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
                make_corsika_steering_card(
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
                make_coreas_steering_card(
                    core_position_on_observation_level_north=(
                        core_position_on_observation_level_north
                    ),
                    core_position_on_observation_level_west=(
                        core_position_on_observation_level_west
                    ),
                    observation_level_altitude=observation_level_altitude,
                    time_slice_duration=time_slice_duration,
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
        os.makedirs(out_event_dir)
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
            corsika_coreas_return_code = subprocess.call(
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


class MyEncoder(json.JSONEncoder):
    """
    By mgilson, Software Engineer at Argo AI, 2017
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


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
    for component in [
        'north_component',
        'west_component',
        'vertical_component'
    ]:
        image_sensor_responses[component] = (
            telescope.simulate_image_sensor_response(
                huygens_matrix=huygens_matrix,
                raw_imaging_reflector_huygens_antenna_responses=(
                    raw_imaging_reflector_huygens_antenna_responses
                ),
                number_time_slices=300,
                component=component
            )
        )

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
            component+'.float32'
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
        fout.write(json.dumps(config, indent=4, cls=MyEncoder))


def sample_zenith_distance(
    min_zenith_distance=np.deg2rad(1),
    max_zenith_distance=np.deg2rad(5),
    size=100
):
    v_min = (np.cos(min_zenith_distance) + 1)/2
    v_max = (np.cos(max_zenith_distance) + 1)/2
    v = np.random.uniform(low=v_min, high=v_max, size=size)
    return np.arccos(2*v - 1)


def sample_2D_points_within_radius(radius, size):
    rho = np.sqrt(np.random.uniform(0, 1, size))*radius
    phi = np.random.uniform(0, 2*np.pi, size)
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
        'event_id': np.arange(1, number_events+1),
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
        )
    }
