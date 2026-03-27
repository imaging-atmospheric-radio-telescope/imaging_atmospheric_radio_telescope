from . import utils as psf_utils
from . import plane_wave_response

from ... import lownoiseblock
from ... import signal
from ... import production
from ... import calibration_source
from ... import calibration
from ... import utils as iaat_utils
from ... import logger as iaat_logger

import os
import numpy as np
import rename_after_writing as rnw
import spherical_coordinates


def make_jobs(work_dir, config):
    PI = np.pi
    jobs = []
    for telescope_key in config["defocus"]["telescopes"]:

        telescope, _, _ = psf_utils.make_telescope_timing_and_site(
            work_dir=work_dir, config=config, telescope_key=telescope_key
        )

        field_of_view_edges = psf_utils.make_field_of_view_region_edges(
            sensor=telescope["sensor"],
            focal_length_m=telescope["mirror"]["focal_length_m"],
        )

        f_m = telescope["mirror"]["focal_length_m"]
        qrng = iaat_utils.QuasiRandomGenerator(seed=123)
        prng = np.random.Generator(np.random.PCG64(123))
        for i in range(config["defocus"]["num"]):
            job = {}
            job["telescope_key"] = telescope_key
            job["id"] = i
            job["sensor_distance_m"] = qrng.uniform(
                low=f_m * config["defocus"]["start_sensor_distance_f"],
                high=f_m * config["defocus"]["stop_sensor_distance_f"],
            )
            job["polarization_angle_rad"] = qrng.uniform(low=-PI, high=PI)

            job["azimuth_rad"], job["zenith_rad"] = (
                spherical_coordinates.random.uniform_az_zd_in_cone(
                    prng=prng,
                    azimuth_rad=0.0,
                    zenith_rad=0.0,
                    min_half_angle_rad=0.0,
                    max_half_angle_rad=(1 / 4)
                    * field_of_view_edges[
                        "field_of_view_fully_inside_half_angle_rad"
                    ],
                )
            )

            job["work_dir"] = work_dir
            job["path"] = os.path.join(
                work_dir, "defocus", job["telescope_key"], f"{job['id']:06d}"
            )
            jobs.append(job)
    return jobs


def drop_finished_jobs(work_dir, jobs):
    out = []
    for job in jobs:
        if not os.path.exists(os.path.join(job["path"])):
            out.append(job)
    return out


def run_job(job):
    config = psf_utils.read_config(job["work_dir"])

    telescope, timing, site = psf_utils.make_telescope_timing_and_site(
        work_dir=job["work_dir"],
        config=config,
        telescope_key=job["telescope_key"],
        sensor_distance_m=job["sensor_distance_m"],
    )

    ecsf = calibration.read_energy_conservation_scale_factor(
        path=os.path.join(
            job["work_dir"],
            "calibration",
            job["telescope_key"],
            "energy_conservation_scale_factor.json",
        )
    )
    mirror_to_camera_energy_scale_factor = ecsf["fitted_energy_scale_factor"]

    source_frequency_Hz = np.mean(
        lownoiseblock.input_frequency_start_stop_Hz(telescope["lnb"])
    )
    wavelength_m = signal.frequency_to_wavelength(source_frequency_Hz)
    num_waves = 7

    region_of_interest_rad = np.arctan(
        (num_waves * wavelength_m) / telescope["mirror"]["focal_length_m"]
    )

    source_config = production.radio_from_plane_wave.make_config()
    s1 = calibration_source.plane_wave_in_far_field.make_config()
    s1["geometry"]["azimuth_rad"] = job["azimuth_rad"]
    s1["geometry"]["zenith_rad"] = job["zenith_rad"]
    s1["geometry"]["polarization_angle_rad"] = job["polarization_angle_rad"]
    s1["sine_wave"]["emission_frequency_Hz"] = source_frequency_Hz
    source_config["plane_waves"] = {}
    source_config["plane_waves"]["1"] = s1

    with rnw.Directory(job["path"]) as tmp_dir:
        logger = iaat_logger.LoggerFile(os.path.join(tmp_dir, "log.jsonl"))

        plane_wave_response.make_PlaneWaveResponse(
            out_dir=tmp_dir,
            random_seed=job["id"],
            telescope=telescope,
            site=site,
            timing=timing,
            mirror_to_camera_energy_scale_factor=mirror_to_camera_energy_scale_factor,
            source_config=source_config,
            region_of_interest=True,
            region_of_interest_rad=region_of_interest_rad,
            region_of_interest_num_bins=psf_utils.substract_one_when_even(
                num_waves * timing["oversampling"]
            ),
            logger=logger,
            save_roi_electric_fields=False,
        )
        response = plane_wave_response.PlaneWaveResponse(tmp_dir)
        response.plot()
