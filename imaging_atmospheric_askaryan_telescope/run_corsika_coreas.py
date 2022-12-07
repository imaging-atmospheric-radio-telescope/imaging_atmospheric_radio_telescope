# Copyright 2017 Sebastian A. Mueller
import numpy as np
import tempfile
import os
import subprocess
import shutil
import json

from . import steering_card_utils
from . import telescope as simtelescope
from . import coreas_bridge
from . import json_numpy_utils as jsonumpy


def estimate_start_time_from_antnna_response(raw_time, raw_field_components):
    num_components = raw_field_components.shape[1]
    max_position_times = np.zeros(num_components)
    for component in range(num_components):
        first_slice = np.min(np.nonzero(raw_field_components[:, component]))
        max_position_times[component] = raw_time[first_slice]
    return np.median(max_position_times)


def simulate_air_shower_and_imaging_reflector_response(
    corsika_coreas_executable_path,
    out_probe_dir,
    out_dir,
    event_id,
    primary_particle_id,
    energy,
    zenith_distance,
    azimuth,
    observation_level_altitude,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
    core_position_on_observation_level_north,
    core_position_on_observation_level_west,
    time_slice_duration,
    mirror_probe_positions,
    time_slice_duration_of_probe=None,
    time_lower_boundary_of_probe=-1000e-6,
    time_upper_boundary_of_probe=25e-6,
    time_window_duration=4e-7,
    fraction_of_time_window_to_be_warm_up_time=0.06,
):
    """
    Simulates the electric-field-strength caused by an air-shower at the
    support-poisitions of the imaging-reflector. The result and log-files of
    the air-shower-simulation are stored in out_dir.

    The time-window for the electric-field-strength is estimated in advance.
    First the air-shower is simulated and the response of one single
    probe-antenna is simulated. Based on the time-series of the probe-antenna,
    the time-window is estimated. Second, the air-shower is simulated again and
    but this time with all support-antennas of the imaging-reflector using the
    time-window estimated in the first step.
    """
    if time_slice_duration_of_probe is None:
        time_slice_duration_of_probe = time_slice_duration

    probe_position = np.zeros(shape=(1, 3))

    os.makedirs(out_probe_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    simulate_air_shower_and_imaging_reflector_response_manual(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_dir=out_probe_dir,
        event_id=event_id,
        primary_particle_id=primary_particle_id,
        energy=energy,
        zenith_distance=zenith_distance,
        azimuth=azimuth,
        core_position_on_observation_level_north=(
            core_position_on_observation_level_north
        ),
        core_position_on_observation_level_west=(
            core_position_on_observation_level_west
        ),
        observation_level_altitude=observation_level_altitude,
        earth_magnetic_field_x_muT=earth_magnetic_field_x_muT,
        earth_magnetic_field_z_muT=earth_magnetic_field_z_muT,
        time_slice_duration=time_slice_duration_of_probe,
        mirror_probe_positions=probe_position,
        coreas_time_boundaries={
            "automatic_time_boundaries": 0,
            "time_lower_boundary": time_lower_boundary_of_probe,
            "time_upper_boundary": time_upper_boundary_of_probe,
        },
    )

    probe_response = coreas_bridge.read_all_raw_time_series(
        os.path.join(out_probe_dir, "antenna_responses")
    )

    start_time = estimate_start_time_from_antnna_response(
        raw_time=probe_response[0, :, 0],
        raw_field_components=probe_response[0, :, 1:4],
    )

    f = fraction_of_time_window_to_be_warm_up_time
    time_lower_boundary = start_time - f * time_window_duration
    time_upper_boundary = start_time + (1 - f) * time_window_duration

    with open(os.path.join(out_probe_dir, "time_window.json"), "wt") as fout:
        fout.write(
            json.dumps(
                {
                    "start_time": start_time,
                    "time_lower_boundary": time_lower_boundary,
                    "time_upper_boundary": time_upper_boundary,
                }
            )
        )

    simulate_air_shower_and_imaging_reflector_response_manual(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_dir=out_dir,
        event_id=event_id,
        primary_particle_id=primary_particle_id,
        energy=energy,
        zenith_distance=zenith_distance,
        azimuth=azimuth,
        core_position_on_observation_level_north=(
            core_position_on_observation_level_north
        ),
        core_position_on_observation_level_west=(
            core_position_on_observation_level_west
        ),
        observation_level_altitude=observation_level_altitude,
        earth_magnetic_field_x_muT=earth_magnetic_field_x_muT,
        earth_magnetic_field_z_muT=earth_magnetic_field_z_muT,
        time_slice_duration=time_slice_duration,
        mirror_probe_positions=mirror_probe_positions,
        coreas_time_boundaries={
            "automatic_time_boundaries": 0,
            "time_lower_boundary": time_lower_boundary,
            "time_upper_boundary": time_upper_boundary,
        },
    )


def simulate_air_shower_and_imaging_reflector_response_manual(
    corsika_coreas_executable_path,
    out_dir,
    event_id,
    primary_particle_id,
    energy,
    zenith_distance,
    azimuth,
    core_position_on_observation_level_north,
    core_position_on_observation_level_west,
    observation_level_altitude,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
    time_slice_duration,
    mirror_probe_positions,
    coreas_time_boundaries=steering_card_utils.DEFAULT_COREAS_TIME_BOUNDARIES,
):
    with tempfile.TemporaryDirectory(prefix="corsika_coreas_") as tmp_dir:
        tmp_run_dir = os.path.join(tmp_dir, "run")

        shutil.copytree(
            os.path.dirname(corsika_coreas_executable_path),
            os.path.abspath(tmp_run_dir),
            symlinks=False,
        )

        # tmp paths
        tmp_corsika_coreas_executable_path = os.path.join(
            tmp_run_dir, os.path.basename(corsika_coreas_executable_path)
        )
        tmp_corsika_steering_card_path = os.path.join(
            tmp_run_dir, "RUN{:06d}.inp".format(event_id)
        )
        tmp_coreas_steering_card_path = os.path.join(
            tmp_run_dir, "SIM{:06d}.reas".format(event_id)
        )
        tmp_coreas_antenna_list_path = os.path.join(
            tmp_run_dir, "SIM{:06d}.list".format(event_id)
        )
        tmp_coreas_raw_antenna_output_dir = os.path.join(
            tmp_run_dir, "SIM{:06d}_coreas".format(event_id)
        )

        with open(tmp_corsika_steering_card_path, "wt") as fout:
            fout.write(
                steering_card_utils.make_corsika_steering_card(
                    event_id=event_id,
                    primary_particle_id=primary_particle_id,
                    energy=energy,
                    zenith_distance=zenith_distance,
                    azimuth=azimuth,
                    observation_level_altitude=observation_level_altitude,
                    earth_magnetic_field_x_muT=earth_magnetic_field_x_muT,
                    earth_magnetic_field_z_muT=earth_magnetic_field_z_muT,
                )
            )

        with open(tmp_coreas_steering_card_path, "wt") as fout:
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
                    time_boundaries=coreas_time_boundaries,
                )
            )

        with open(tmp_coreas_antenna_list_path, "wt") as fout:
            fout.write(
                steering_card_utils.make_coreas_antenna_list(
                    positions=mirror_probe_positions
                )
            )

        corsika_steering_card_pipe, pwrite = os.pipe()
        with open(tmp_corsika_steering_card_path, "rt") as fin:
            os.write(pwrite, str.encode(fin.read()))
            os.close(pwrite)

        # output paths
        os.makedirs(out_dir, exist_ok=True)
        out_corsika_coreas_dir = os.path.join(out_dir, "corsika_coreas")
        os.makedirs(out_corsika_coreas_dir)

        out_corsika_o_path = os.path.join(out_corsika_coreas_dir, "corsika.o")
        out_corsika_e_path = os.path.join(out_corsika_coreas_dir, "corsika.e")
        out_antenna_output_dir = os.path.join(out_dir, "antenna_responses")

        with open(out_corsika_o_path, "w") as corsika_o, open(
            out_corsika_e_path, "w"
        ) as corsika_e:
            subprocess.call(
                tmp_corsika_coreas_executable_path,
                stdin=corsika_steering_card_pipe,
                stdout=corsika_o,
                stderr=corsika_e,
                cwd=tmp_run_dir,
            )

        shutil.move(tmp_coreas_raw_antenna_output_dir, out_antenna_output_dir)
        shutil.move(
            tmp_coreas_antenna_list_path,
            os.path.join(
                out_corsika_coreas_dir,
                os.path.basename(tmp_coreas_antenna_list_path),
            ),
        )
        shutil.move(
            tmp_coreas_steering_card_path,
            os.path.join(
                out_corsika_coreas_dir,
                os.path.basename(tmp_coreas_steering_card_path),
            ),
        )
        shutil.move(
            tmp_corsika_steering_card_path,
            os.path.join(
                out_corsika_coreas_dir,
                os.path.basename(tmp_corsika_steering_card_path),
            ),
        )

        # input('wait to inspect the tmp directory')


def simulate_event(
    corsika_coreas_executable_path,
    out_dir,
    event_id,
    primary_particle_id,
    energy,
    zenith_distance,
    azimuth,
    core_position_on_observation_level_north,
    core_position_on_observation_level_west,
    observation_level_altitude,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
    time_slice_duration,
    telescope,
    num_time_slices=300,
):
    """
    Does a full simulation of a single event from the shower to the sensor
    response.
    Corsika -> Coreas -> Reflector -> Sensor Response.
    Output will be written into out_dir.
    """
    simulate_air_shower_and_imaging_reflector_response(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_probe_dir=os.path.join(out_dir, "time_window"),
        out_dir=os.path.join(out_dir, "mirror"),
        event_id=event_id,
        primary_particle_id=primary_particle_id,
        energy=energy,
        zenith_distance=zenith_distance,
        azimuth=azimuth,
        observation_level_altitude=observation_level_altitude,
        earth_magnetic_field_x_muT=earth_magnetic_field_x_muT,
        earth_magnetic_field_z_muT=earth_magnetic_field_z_muT,
        core_position_on_observation_level_north=(
            core_position_on_observation_level_north
        ),
        core_position_on_observation_level_west=(
            core_position_on_observation_level_west
        ),
        time_slice_duration=time_slice_duration,
        mirror_probe_positions=telescope["mirror"]["probe_positions"],
    )

    mirror_antenna_responses = coreas_bridge.read_electric_field_on_imaging_reflector(
        path=os.path.join(out_dir, "mirror", "antenna_responses")
    )

    coreas_bridge.write_antenna_response(
        response=mirror_antenna_responses,
        path=os.path.join(out_dir, "mirror", "antenna_responses_bin"),
    )

    sensor_responses = {}
    components = ["north", "west", "vertical"]
    for component in components:
        sensor_response = simtelescope.make_feed_horn_responses(
            telescope=telescope,
            mirror_antenna_responses=mirror_antenna_responses,
            num_time_slices=num_time_slices,
            component=component,
        )
        sensor_responses[component] = sensor_response

    sensor_dir = os.path.join(out_dir, "sensor")
    os.makedirs(sensor_dir)

    for component in components:
        out_component_path = os.path.join(
            sensor_dir, "electric_field." + component + ".float32"
        )
        with open(out_component_path, "wb") as fout:
            fout.write(np.float32(sensor_responses[component]).tobytes())

    with open(
        os.path.join(sensor_dir, "time_slice_duration.float64"), "wb"
    ) as fout:
        fout.write(np.float64(time_slice_duration).tobytes())

    with open(
        os.path.join(sensor_dir, "num_time_slices.uint64"), "wb"
    ) as fout:
        fout.write(np.uint64(num_time_slices).tobytes())
