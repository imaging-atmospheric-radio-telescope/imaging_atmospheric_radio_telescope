# Copyright 2017 Sebastian A. Mueller
import numpy as np
import tempfile
import os
import subprocess
import shutil
import json

from .. import telescope as simtelescope
from .. import electric_fields
from .. import timing_and_sampling
from .. import corsika


def simulate_mirror_electric_fields_manual(
    out_dir,
    event_id,
    primary_particle,
    site,
    time_slice_duration_s,
    antenna_positions_obslvl_m,
    coreas_time_boundaries=corsika.coreas.DEFAULT_TIME_BOUNDARIES,
    corsika_coreas_executable_path=None,
):
    if corsika_coreas_executable_path is None:
        corsika_coreas_executable_path = (
            corsika.install.get_corsika_executable_path()
        )

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
                    unique_identifier=event_id,
                    primary_particle_key=primary_particle["key"],
                    energy_GeV=primary_particle["energy_GeV"],
                    zenith_distance_deg=primary_particle[
                        "zenith_distance_deg"
                    ],
                    azimuth_deg=primary_particle["azimuth_deg"],
                    observation_level_asl_m=site["observation_level_asl_m"],
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
                    core_position_on_observation_level_north_m=primary_particle[
                        "core_north_m"
                    ],
                    core_position_on_observation_level_west_m=primary_particle[
                        "core_west_m"
                    ],
                    core_position_on_observation_level_asl_m=site[
                        "observation_level_asl_m"
                    ],
                    time_slice_duration_s=time_slice_duration_s,
                    time_boundaries=coreas_time_boundaries,
                )
            )

        antenna_positions_asl_m = antenna_positions_obslvl_m.copy()
        antenna_positions_asl_m[:, 2] += site["observation_level_asl_m"]

        with open(tmp_coreas_antenna_list_path, "wt") as fout:
            fout.write(
                corsika.coreas.make_antenna_list(
                    positions_asl_m=antenna_positions_asl_m
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

        raw_antenna_dir = os.path.join(out_dir, "electric_fields.raw")
        antenna_path = os.path.join(out_dir, "electric_fields.tar")

        with open(cor_o_path, "w") as cor_o, open(cor_e_path, "w") as cor_e:
            subprocess.call(
                tmp_corsika_coreas_executable_path,
                stdin=cor_i,
                stdout=cor_o,
                stderr=cor_e,
                cwd=tmp_run_dir,
            )

        # input cards
        shutil.move(
            tmp_coreas_antenna_list_path,
            os.path.join(
                cor_dir,
                os.path.basename(tmp_coreas_antenna_list_path),
            ),
        )
        shutil.move(
            tmp_coreas_steering_card_path,
            os.path.join(
                cor_dir,
                os.path.basename(tmp_coreas_steering_card_path),
            ),
        )
        shutil.move(
            tmp_corsika_steering_card_path,
            os.path.join(
                cor_dir,
                os.path.basename(tmp_corsika_steering_card_path),
            ),
        )

        # output electric fields
        shutil.move(tmp_coreas_antenna_dir, raw_antenna_dir)

        # unify output antennas
        raw_electric_fields = corsika.coreas.read_raw_electric_fields(
            raw_antenna_dir
        )
        unified_electric_field = corsika.coreas.make_electric_fields(
            raw_electric_fields=raw_electric_fields
        )
        electric_fields.write_tar(
            path=antenna_path,
            electric_fields=unified_electric_field,
        )

        # input('wait to inspect the tmp directory')


def simulate_telescope_response(
    out_dir,
    event_id,
    primary_particle,
    site,
    telescope,
    timing,
    corsika_coreas_executable_path=None,
):
    """
    Does a full simulation of a single event from the shower to the sensor
    response.
    Corsika -> Coreas -> Reflector -> Sensor Response.
    Output will be written into out_dir.
    """

    probe_dir = os.path.join(out_dir, "probe")
    if not os.path.exists(probe_dir):

        start_time_probe = timing["start_time_probe"]
        simulate_mirror_electric_fields_manual(
            corsika_coreas_executable_path=corsika_coreas_executable_path,
            out_dir=probe_dir,
            event_id=event_id,
            primary_particle=primary_particle,
            site=site,
            time_slice_duration_s=start_time_probe["time_slice_duration_s"],
            antenna_positions_obslvl_m=np.array(
                [start_time_probe["position_m"]]
            ),
            coreas_time_boundaries={
                "automatic_time_boundaries_s": 0,
                "time_lower_boundary_s": start_time_probe[
                    "time_lower_boundary_s"
                ],
                "time_upper_boundary_s": start_time_probe[
                    "time_upper_boundary_s"
                ],
            },
        )

        probe_electric_fields = electric_fields.read_tar(
            path=os.path.join(probe_dir, "electric_fields.tar")
        )

        start_time_based_on_probe = (
            timing_and_sampling.estimate_start_time_from_electric_fields(
                electric_fields=probe_electric_fields
            )
        )

        (
            time_lower_boundary_s,
            time_upper_boundary_s,
        ) = timing_and_sampling.make_time_window_bounds(
            start_time_s=start_time_based_on_probe,
            time_window_duration_s=timing["electric_fields"]["mirror"][
                "time_window_duration_s"
            ],
            fraction_of_time_window_to_be_warm_up_time=timing[
                "electric_fields"
            ]["mirror"]["warm_up_fraction_wrt_to_start_time_probe"],
        )

        time_window = {
            "start_time_based_on_probe_s": start_time_based_on_probe,
            "time_lower_boundary_s": time_lower_boundary_s,
            "time_upper_boundary_s": time_upper_boundary_s,
        }

        with open(os.path.join(out_dir, "time_window.json"), "wt") as fout:
            fout.write(json.dumps(time_window, indent=4))

    mirror_dir = os.path.join(out_dir, "mirror")
    if not os.path.exists(mirror_dir):

        with open(os.path.join(out_dir, "time_window.json"), "rt") as f:
            time_window = json.loads(f.read())

        simulate_mirror_electric_fields_manual(
            corsika_coreas_executable_path=corsika_coreas_executable_path,
            out_dir=mirror_dir,
            event_id=event_id,
            primary_particle=primary_particle,
            site=site,
            time_slice_duration_s=timing["electric_fields"][
                "time_slice_duration_s"
            ],
            antenna_positions_obslvl_m=telescope["mirror"][
                "scatter_center_positions_m"
            ],
            coreas_time_boundaries={
                "automatic_time_boundaries_s": 0,
                "time_lower_boundary_s": time_window["time_lower_boundary_s"],
                "time_upper_boundary_s": time_window["time_upper_boundary_s"],
            },
        )

    sensor_dir = os.path.join(out_dir, "sensor")
    if not os.path.exists(sensor_dir):
        os.makedirs(sensor_dir)

        mirror_electric_fields = electric_fields.read_tar(
            path=os.path.join(out_dir, "mirror", "electric_fields.tar"),
        )

        sensor_electric_fields = (
            simtelescope.propagate_electric_field_from_mirror_to_sensor(
                telescope=telescope,
                mirror_electric_fields=mirror_electric_fields,
                num_time_slices=timing["electric_fields"]["sensor"][
                    "num_time_slices"
                ],
            )
        )

        electric_fields.write_tar(
            path=os.path.join(sensor_dir, "electric_fields.tar"),
            electric_fields=sensor_electric_fields,
        )

    lnb_dir = os.path.join(out_dir, "lnb")
    if not os.path.exists(lnb_dir):
        pass
