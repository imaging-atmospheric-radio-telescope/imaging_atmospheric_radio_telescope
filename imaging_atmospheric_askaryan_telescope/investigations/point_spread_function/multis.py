from . import utils as psf_utils
from . import plane_wave_response

from ... import lownoiseblock
from ... import signal
from ... import production
from ... import calibration_source
from ... import logger as iaat_logger

import os
import numpy as np
import rename_after_writing as rnw
import spherical_coordinates


def make_jobs(work_dir, config):
    prng = np.random.Generator(
        np.random.PCG64(config["multis"]["random_seed"])
    )

    jobs = []
    for telescope_key in config["multis"]["telescopes"]:

        telescope, _, _ = psf_utils.make_telescope_timing_and_site(
            config=config, telescope_key=telescope_key
        )

        for i in range(config["multis"]["num"]):
            job = {}
            job["telescope_key"] = telescope_key
            job["id"] = i
            job["work_dir"] = work_dir
            job["path"] = os.path.join(
                work_dir, "multis", job["telescope_key"], f"{job['id']:06d}"
            )
            job["sources"] = {}
            for i in range(config["multis"]["num_sources_per_event"]):
                s = _draw_source(prng=prng, config=config, telescope=telescope)
                job["sources"][f"{i:d}"] = s

            jobs.append(job)
    return jobs


def _draw_source(prng, config, telescope):
    field_of_view_edges = psf_utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )

    telescope_nu_start_Hz, telescope_nu_stop_Hz = (
        lownoiseblock.input_frequency_start_stop_Hz(lnb=telescope["lnb"])
    )

    job = {}
    job["azimuth_rad"], job["zenith_rad"] = (
        spherical_coordinates.random.uniform_az_zd_in_cone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_half_angle_rad=0.0,
            max_half_angle_rad=field_of_view_edges[
                "field_of_view_fully_inside_half_angle_rad"
            ],
        )
    )
    job["frequency_Hz"] = prng.uniform(
        low=telescope_nu_start_Hz,
        high=telescope_nu_stop_Hz,
    )
    job["power_density_W_per_m2"] = prng.uniform(
        low=config["multis"]["power_density_start_W_per_m2"],
        high=config["multis"]["power_density_stop_W_per_m2"],
    )
    job["polarization_angle_rad"] = prng.uniform(low=-np.pi, high=np.pi)
    return job


def drop_finished_jobs(work_dir, jobs):
    out = []
    for job in jobs:
        if not os.path.exists(os.path.join(job["path"])):
            out.append(job)
    return out


def run_job(job):
    config = psf_utils.read_config(job["work_dir"])

    telescope, timing, site = psf_utils.make_telescope_timing_and_site(
        config=config, telescope_key=job["telescope_key"]
    )

    source_config = production.radio_from_plane_wave.make_config()
    source_config["plane_waves"] = {}
    for key in job["sources"]:
        _s = job["sources"][key]
        s = calibration_source.plane_wave_in_far_field.make_config()
        s["geometry"]["azimuth_rad"] = _s["azimuth_rad"]
        s["geometry"]["zenith_rad"] = _s["zenith_rad"]
        s["geometry"]["polarization_angle_rad"] = _s["polarization_angle_rad"]
        s = psf_utils.set_power_with_areal_density(
            plane_wave_config=s,
            power_density_W_per_m2=_s["power_density_W_per_m2"],
        )
        s["sine_wave"]["emission_frequency_Hz"] = _s["frequency_Hz"]
        source_config["plane_waves"][key] = s

    with rnw.Directory(job["path"]) as tmp_dir:
        logger = iaat_logger.LoggerFile(os.path.join(tmp_dir, "log.jsonl"))

        plane_wave_response.make_PlaneWaveResponse(
            out_dir=tmp_dir,
            random_seed=job["id"],
            telescope=telescope,
            site=site,
            timing=timing,
            source_config=source_config,
            region_of_interest=False,
            region_of_interest_rad=None,
            region_of_interest_num_bins=None,
            logger=logger,
            save_roi_electric_fields=False,
        )
        response = plane_wave_response.PlaneWaveResponse(tmp_dir)
        response.plot()
