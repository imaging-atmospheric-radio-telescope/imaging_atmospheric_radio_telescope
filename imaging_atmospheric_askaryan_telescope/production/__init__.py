# Copyright 2017 Sebastian A. Mueller
import numpy as np
import tempfile
import os
import subprocess
import shutil
import json

from .. import telescope as simtelescope
from .. import corsika
from .. import json_numpy_utils as jsonumpy
from . import time_window


def simulate_mirror_electric_fields(
    corsika_coreas_executable_path,
    out_probe_dir,
    out_dir,
    event_id,
    primary_particle,
    site,
    time_slice_duration,
    time_window_duration,
    mirror_antenna_positions,
    time_slice_duration_of_probe=None,
    time_lower_boundary_of_probe=-1000e-6,
    time_upper_boundary_of_probe=25e-6,
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

    probe_position = np.array([[0.0, 0.0, 0.0]])

    os.makedirs(out_probe_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    simulate_mirror_electric_fields_manual(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_dir=out_probe_dir,
        event_id=event_id,
        primary_particle=primary_particle,
        site=site,
        time_slice_duration=time_slice_duration_of_probe,
        mirror_antenna_positions=probe_position,
        coreas_time_boundaries={
            "automatic_time_boundaries": 0,
            "time_lower_boundary": time_lower_boundary_of_probe,
            "time_upper_boundary": time_upper_boundary_of_probe,
        },
    )

    probe_electric_field = corsika.coreas.read_raw_electric_fields(
        os.path.join(out_probe_dir, "electric_fields")
    )

    start_time = time_window.estimate_start_time_from_antnna_response(
        raw_time=probe_electric_field[0, :, 0],
        raw_field_components=probe_electric_field[0, :, 1:4],
    )

    (
        time_lower_boundary,
        time_upper_boundary,
    ) = time_window.make_time_window_bounds(
        start_time=start_time,
        time_window_duration=time_window_duration,
        fraction_of_time_window_to_be_warm_up_time=fraction_of_time_window_to_be_warm_up_time,
    )

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

    simulate_mirror_electric_fields_manual(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_dir=out_dir,
        event_id=event_id,
        primary_particle=primary_particle,
        site=site,
        time_slice_duration=time_slice_duration,
        mirror_antenna_positions=mirror_antenna_positions,
        coreas_time_boundaries={
            "automatic_time_boundaries": 0,
            "time_lower_boundary": time_lower_boundary,
            "time_upper_boundary": time_upper_boundary,
        },
    )


def simulate_mirror_electric_fields_manual(
    corsika_coreas_executable_path,
    out_dir,
    event_id,
    primary_particle,
    site,
    time_slice_duration,
    mirror_antenna_positions,
    coreas_time_boundaries=corsika.coreas.DEFAULT_TIME_BOUNDARIES,
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
        tmp_coreas_antenna_dir = os.path.join(
            tmp_run_dir, "SIM{:06d}_coreas".format(event_id)
        )

        with open(tmp_corsika_steering_card_path, "wt") as fout:
            fout.write(
                corsika.make_steering_card(
                    event_id=event_id,
                    primary_particle_id=primary_particle["id"],
                    energy=primary_particle["energy_GeV"],
                    zenith_distance=primary_particle["zenith_distance_rad"],
                    azimuth=primary_particle["azimuth_rad"],
                    observation_level_altitude=site[
                        "observation_level_altitude"
                    ],
                    earth_magnetic_field_x_muT=site[
                        "earth_magnetic_field_x_muT"
                    ],
                    earth_magnetic_field_z_muT=site[
                        "earth_magnetic_field_z_muT"
                    ],
                )
            )

        with open(tmp_coreas_steering_card_path, "wt") as fout:
            fout.write(
                corsika.coreas.make_steering_card(
                    core_position_on_observation_level_north=primary_particle[
                        "core_north_m"
                    ],
                    core_position_on_observation_level_west=primary_particle[
                        "core_west_m"
                    ],
                    observation_level_altitude=site[
                        "observation_level_altitude"
                    ],
                    time_slice_duration=time_slice_duration,
                    time_boundaries=coreas_time_boundaries,
                )
            )

        with open(tmp_coreas_antenna_list_path, "wt") as fout:
            fout.write(
                corsika.coreas.make_antenna_list(
                    positions=mirror_antenna_positions
                )
            )

        cor_i, pwrite = os.pipe()
        with open(tmp_corsika_steering_card_path, "rt") as fin:
            os.write(pwrite, str.encode(fin.read()))
            os.close(pwrite)

        # output paths
        os.makedirs(out_dir, exist_ok=True)
        cor_dir = os.path.join(out_dir, "corsika_coreas")
        os.makedirs(cor_dir)

        cor_o_path = os.path.join(cor_dir, "corsika.o")
        cor_e_path = os.path.join(cor_dir, "corsika.e")
        antenna_dir = os.path.join(out_dir, "electric_fields")

        with open(cor_o_path, "w") as cor_o, open(cor_e_path, "w") as cor_e:
            subprocess.call(
                tmp_corsika_coreas_executable_path,
                stdin=cor_i,
                stdout=cor_o,
                stderr=cor_e,
                cwd=tmp_run_dir,
            )

        shutil.move(tmp_coreas_antenna_dir, antenna_dir)
        shutil.move(
            tmp_coreas_antenna_list_path,
            os.path.join(
                cor_dir, os.path.basename(tmp_coreas_antenna_list_path),
            ),
        )
        shutil.move(
            tmp_coreas_steering_card_path,
            os.path.join(
                cor_dir, os.path.basename(tmp_coreas_steering_card_path),
            ),
        )
        shutil.move(
            tmp_corsika_steering_card_path,
            os.path.join(
                cor_dir, os.path.basename(tmp_corsika_steering_card_path),
            ),
        )

        # input('wait to inspect the tmp directory')


def simulate_telescope_response(
    corsika_coreas_executable_path,
    out_dir,
    event_id,
    primary_particle,
    site,
    time_slice_duration,
    time_window_duration,
    telescope,
    num_time_slices=300,
):
    """
    Does a full simulation of a single event from the shower to the sensor
    response.
    Corsika -> Coreas -> Reflector -> Sensor Response.
    Output will be written into out_dir.
    """

    if not os.path.exists(os.path.join(out_dir, "mirror")):
        simulate_mirror_electric_fields(
            corsika_coreas_executable_path=corsika_coreas_executable_path,
            out_probe_dir=os.path.join(out_dir, "time_window"),
            out_dir=os.path.join(out_dir, "mirror"),
            event_id=event_id,
            primary_particle=primary_particle,
            site=site,
            time_slice_duration=time_slice_duration,
            time_window_duration=time_window_duration,
            mirror_antenna_positions=telescope["mirror"]["antenna_positions"],
        )

        mirror_raw_electric_fields = corsika.coreas.read_raw_electric_fields(
            path=os.path.join(out_dir, "mirror", "electric_fields"),
        )

        mirror_electric_fields = corsika.coreas.make_electric_fields(
            raw_electric_fields=mirror_raw_electric_fields
        )

        simtelescope.write_electric_fields(
            path=os.path.join(out_dir, "mirror", "electric_fields"),
            electric_fields=mirror_electric_fields,
        )

    sensor_dir = os.path.join(out_dir, "sensor")
    sensor_electric_fields_dir = os.path.join(sensor_dir, "electric_fields")

    if not os.path.exists(sensor_electric_fields_dir):
        os.makedirs(sensor_electric_fields_dir)

        mirror_electric_fields = simtelescope.read_electric_fields(
            path=os.path.join(out_dir, "mirror", "electric_fields"),
        )

        sensor_electric_fields = simtelescope.make_sensor_electric_fields(
            telescope=telescope,
            mirror_electric_fields=mirror_electric_fields,
            num_time_slices=num_time_slices,
        )

        simtelescope.write_electric_fields(
            path=sensor_electric_fields_dir,
            electric_fields=sensor_electric_fields,
        )
