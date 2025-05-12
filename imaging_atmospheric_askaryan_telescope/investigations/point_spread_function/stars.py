from . import utils as psf_utils
from . import plane_wave_response

from ... import lownoiseblock
from ... import signal
from ... import production
from ... import calibration_source

import os
import numpy as np
import spherical_coordinates
import rename_after_writing as rnw
import json_line_logger


def make_jobs(work_dir, config):
    jobs = []
    for telescope_key in config["stars"]["telescopes"]:

        telescope, _, _ = psf_utils.make_telescope_timing_and_site(
            config=config, telescope_key=telescope_key
        )

        jobs += _make_jobs_representative_guide_stars(
            work_dir=work_dir,
            config=config,
            telescope=telescope,
        )

        jobs += _make_jobs_central_feed_horn_scan(
            work_dir=work_dir,
            config=config,
            telescope=telescope,
        )

        jobs += _make_jobs_fully_inside_field_of_view(
            work_dir=work_dir,
            config=config,
            telescope=telescope,
        )

        jobs += _make_jobs_on_edge_of_field_of_view(
            work_dir=work_dir,
            config=config,
            telescope=telescope,
        )

        jobs += _make_jobs_on_edge_of_field_of_view(
            work_dir=work_dir,
            config=config,
            telescope=telescope,
        )

    return jobs


def _make_jobs_representative_guide_stars(work_dir, config, telescope):
    sckey = "representative_guide_stars"
    prng = np.random.Generator(
        np.random.PCG64(config["stars"]["scenarios"][sckey]["random_seed"])
    )

    field_of_view_edges = psf_utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )

    # =======
    field_of_view_scan_zenith_rad = np.linspace(
        0.0,
        field_of_view_edges["field_of_view_fully_inside_half_angle_rad"],
        config["stars"]["scenarios"][sckey]["num"],
    )
    jobs = []
    for i in range(config["stars"]["scenarios"][sckey]["num"]):
        job = {}
        job["key"] = sckey
        job["id"] = i
        job["region_of_interest"] = True
        job["source_azimuth_rad"] = 0.0
        job["source_zenith_rad"] = field_of_view_scan_zenith_rad[i]
        jobs.append(job)

    jobs = _finish_jobs(
        work_dir=work_dir,
        config=config,
        telescope=telescope,
        jobs=jobs,
        prng=prng,
    )
    return jobs


def _make_jobs_central_feed_horn_scan(work_dir, config, telescope):
    sckey = "central_feed_horn_scan"
    prng = np.random.Generator(
        np.random.PCG64(config["stars"]["scenarios"][sckey]["random_seed"])
    )

    field_of_view_edges = psf_utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )

    jobs = []
    for i in range(config["stars"]["scenarios"][sckey]["num"]):
        job = {}
        job["key"] = sckey
        job["id"] = i
        job["region_of_interest"] = False
        az_rad, zd_rad = spherical_coordinates.random.uniform_az_zd_in_cone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_half_angle_rad=0.0,
            max_half_angle_rad=4.0
            * field_of_view_edges["central_feed_horn_half_angle_rad"],
        )
        job["source_azimuth_rad"] = az_rad
        job["source_zenith_rad"] = zd_rad
        jobs.append(job)

    jobs = _finish_jobs(
        work_dir=work_dir,
        config=config,
        telescope=telescope,
        jobs=jobs,
        prng=prng,
    )
    return jobs


def _make_jobs_fully_inside_field_of_view(work_dir, config, telescope):
    sckey = "fully_inside_field_of_view"
    prng = np.random.Generator(
        np.random.PCG64(config["stars"]["scenarios"][sckey]["random_seed"])
    )

    field_of_view_edges = psf_utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )

    jobs = []
    for i in range(config["stars"]["scenarios"][sckey]["num"]):
        job = {}
        job["key"] = sckey
        job["id"] = i
        job["region_of_interest"] = False
        az_rad, zd_rad = spherical_coordinates.random.uniform_az_zd_in_cone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_half_angle_rad=0.0,
            max_half_angle_rad=field_of_view_edges[
                "field_of_view_fully_inside_half_angle_rad"
            ],
        )
        job["source_azimuth_rad"] = az_rad
        job["source_zenith_rad"] = zd_rad
        jobs.append(job)

    jobs = _finish_jobs(
        work_dir=work_dir,
        config=config,
        telescope=telescope,
        jobs=jobs,
        prng=prng,
    )
    return jobs


def _make_jobs_on_edge_of_field_of_view(work_dir, config, telescope):
    sckey = "on_edge_of_field_of_view"
    prng = np.random.Generator(
        np.random.PCG64(config["stars"]["scenarios"][sckey]["random_seed"])
    )
    field_of_view_edges = psf_utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    jobs = []
    for i in range(config["stars"]["scenarios"][sckey]["num"]):
        job = {}
        job["key"] = sckey
        job["id"] = i
        job["region_of_interest"] = False
        az_rad, zd_rad = spherical_coordinates.random.uniform_az_zd_in_cone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_half_angle_rad=field_of_view_edges[
                "field_of_view_fully_inside_half_angle_rad"
            ],
            max_half_angle_rad=field_of_view_edges[
                "field_of_view_fully_outside_half_angle_rad"
            ],
        )
        job["source_azimuth_rad"] = az_rad
        job["source_zenith_rad"] = zd_rad
        jobs.append(job)

    jobs = _finish_jobs(
        work_dir=work_dir,
        config=config,
        telescope=telescope,
        jobs=jobs,
        prng=prng,
    )
    return jobs


def _make_jobs_on_edge_of_field_of_view(work_dir, config, telescope):
    sckey = "on_edge_of_field_of_view"
    prng = np.random.Generator(
        np.random.PCG64(config["stars"]["scenarios"][sckey]["random_seed"])
    )
    field_of_view_edges = psf_utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    jobs = []
    for i in range(config["stars"]["scenarios"][sckey]["num"]):
        job = {}
        job["key"] = sckey
        job["id"] = i
        job["region_of_interest"] = False
        az_rad, zd_rad = spherical_coordinates.random.uniform_az_zd_in_cone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_half_angle_rad=field_of_view_edges[
                "field_of_view_fully_outside_half_angle_rad"
            ],
            max_half_angle_rad=4.0
            * field_of_view_edges[
                "field_of_view_fully_outside_half_angle_rad"
            ],
        )
        job["source_azimuth_rad"] = az_rad
        job["source_zenith_rad"] = zd_rad
        jobs.append(job)

    jobs = _finish_jobs(
        work_dir=work_dir,
        config=config,
        telescope=telescope,
        jobs=jobs,
        prng=prng,
    )
    return jobs


def _finish_jobs(work_dir, config, telescope, jobs, prng):
    telescope_nu_start_Hz, telescope_nu_stop_Hz = (
        lownoiseblock.input_frequency_start_stop_Hz(lnb=telescope["lnb"])
    )

    for job in jobs:
        job["telescope_key"] = telescope["key"]
        job["path"] = os.path.join(
            work_dir,
            "stars",
            job["telescope_key"],
            job["key"],
            f"{job['id']:06d}",
        )
        job["frequency_Hz"] = prng.uniform(
            low=telescope_nu_start_Hz,
            high=telescope_nu_stop_Hz,
        )

        job["work_dir"] = work_dir
        job["source_polarization_angle_rad"] = prng.uniform(
            low=0.0, high=2.0 * np.pi
        )
        job["power_density_W_per_m2"] = prng.uniform(
            low=config["stars"]["power_density_start_W_per_m2"],
            high=config["stars"]["power_density_stop_W_per_m2"],
        )
    return jobs


def drop_finished_jobs(work_dir, jobs):
    out = []
    for job in jobs:
        if not os.path.exists(os.path.join(job["path"])):
            out.append(job)
    return out


def run_job(job):
    config = psf_utils.read_config(job["work_dir"])

    tscope, timing, site = psf_utils.make_telescope_timing_and_site(
        config=config, telescope_key=job["telescope_key"]
    )

    nu_Hz = np.mean(lownoiseblock.input_frequency_start_stop_Hz(tscope["lnb"]))
    wavelength_m = signal.frequency_to_wavelength(nu_Hz)
    num_waves = 7

    region_of_interest_rad = np.arctan(
        (num_waves * wavelength_m) / tscope["mirror"]["focal_length_m"]
    )

    r_100km = 100e3
    A_sphere_100km = 4.0 * np.pi * r_100km**2
    P_isotrop_100km_W = job["power_density_W_per_m2"] * A_sphere_100km

    source_config = production.radio_from_plane_wave.make_config()
    s1 = calibration_source.plane_wave_in_far_field.make_config()
    s1["geometry"]["azimuth_rad"] = job["source_azimuth_rad"]
    s1["geometry"]["zenith_rad"] = job["source_zenith_rad"]
    s1["geometry"]["polarization_angle_rad"] = job[
        "source_polarization_angle_rad"
    ]
    s1["power"][
        "power_of_isotrop_and_point_like_emitter_W"
    ] = P_isotrop_100km_W
    s1["power"]["distance_to_isotrop_and_point_like_emitter_m"] = r_100km
    s1["sine_wave"]["emission_frequency_Hz"] = job["frequency_Hz"]
    source_config["plane_waves"] = {}
    source_config["plane_waves"]["1"] = s1

    with rnw.Directory(job["path"]) as tmp_dir:
        logger = json_line_logger.LoggerFile(
            os.path.join(tmp_dir, "log.jsonl")
        )

        plane_wave_response.make_PlaneWaveResponse(
            out_dir=tmp_dir,
            random_seed=job["id"],
            telescope=tscope,
            site=site,
            timing=timing,
            source_config=source_config,
            region_of_interest=job["region_of_interest"],
            region_of_interest_rad=region_of_interest_rad,
            region_of_interest_num_bins=psf_utils.substract_one_when_even(
                num_waves * timing["oversampling"]
            ),
            logger=logger,
        )
