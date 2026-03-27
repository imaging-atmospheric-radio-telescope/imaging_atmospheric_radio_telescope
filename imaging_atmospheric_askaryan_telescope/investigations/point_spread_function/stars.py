from . import utils as psf_utils
from . import plane_wave_response

from ... import lownoiseblock
from ... import signal
from ... import production
from ... import calibration
from ... import electric_fields
from ... import calibration_source
from ... import logger as iaat_logger
from ... import utils as iaat_utils

import os
import glob
import numpy as np
import spherical_coordinates
import rename_after_writing as rnw


def make_jobs_which_need_energy_calibration(work_dir, config):
    jobs = []
    for telescope_key in config["stars"]["telescopes"]:

        telescope, _, _ = psf_utils.make_telescope_timing_and_site(
            work_dir=work_dir, config=config, telescope_key=telescope_key
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

        jobs += _make_jobs_fully_outside_field_of_view(
            work_dir=work_dir,
            config=config,
            telescope=telescope,
        )

    return jobs


def make_jobs_for_energy_calibration(work_dir, config):
    jobs = []
    for telescope_key in config["stars"]["telescopes"]:

        telescope, _, _ = psf_utils.make_telescope_timing_and_site(
            work_dir=work_dir, config=config, telescope_key=telescope_key
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
        job["apply_mirror_to_camera_energy_scale_factor"] = True
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
        fix_frequency=True,
        fix_power_density=True,
        fix_polarization_angle=True,
    )
    return jobs


def _make_jobs_central_feed_horn_scan(work_dir, config, telescope):
    sckey = "central_feed_horn_scan"
    qrng = iaat_utils.QuasiRandomGenerator(
        seed=config["stars"]["scenarios"][sckey]["random_seed"]
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
        job["apply_mirror_to_camera_energy_scale_factor"] = True
        job["region_of_interest"] = False
        # This is not uniform in solid angle on purpose!
        # Distribution will be uniform in zenith angle what will
        # lead to a cluster near zenith.
        az_rad = qrng.uniform(low=-np.pi, high=np.pi)
        zd_rad = qrng.uniform(
            low=0.0,
            high=4.0 * field_of_view_edges["central_feed_horn_half_angle_rad"],
        )
        job["source_azimuth_rad"] = az_rad
        job["source_zenith_rad"] = zd_rad
        jobs.append(job)

    jobs = _finish_jobs(
        work_dir=work_dir,
        config=config,
        telescope=telescope,
        jobs=jobs,
        prng=qrng,
        fix_frequency=True,
        fix_power_density=True,
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
        job["apply_mirror_to_camera_energy_scale_factor"] = False
        job["region_of_interest"] = True
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
        job["apply_mirror_to_camera_energy_scale_factor"] = False
        job["region_of_interest"] = True
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


def _make_jobs_fully_outside_field_of_view(work_dir, config, telescope):
    sckey = "fully_outside_field_of_view"
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
        job["apply_mirror_to_camera_energy_scale_factor"] = True
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


def _finish_jobs(
    work_dir,
    config,
    telescope,
    jobs,
    prng,
    fix_polarization_angle=False,
    fix_frequency=False,
    fix_power_density=False,
):
    PI = np.pi
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
        if fix_frequency:
            job["frequency_Hz"] = np.mean(
                [telescope_nu_start_Hz, telescope_nu_stop_Hz]
            )
        else:
            job["frequency_Hz"] = prng.uniform(
                low=telescope_nu_start_Hz,
                high=telescope_nu_stop_Hz,
            )

        job["work_dir"] = work_dir

        if fix_polarization_angle:
            job["source_polarization_angle_rad"] = 0.0
        else:
            job["source_polarization_angle_rad"] = prng.uniform(
                low=-PI, high=PI
            )

        if fix_power_density:
            job["power_density_W_per_m2"] = np.mean(
                [
                    config["stars"]["power_density_start_W_per_m2"],
                    config["stars"]["power_density_stop_W_per_m2"],
                ]
            )
        else:
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
        work_dir=job["work_dir"],
        config=config,
        telescope_key=job["telescope_key"],
    )

    if job["apply_mirror_to_camera_energy_scale_factor"]:
        ecsf = calibration.read_energy_conservation_scale_factor(
            path=os.path.join(
                job["work_dir"],
                "calibration",
                job["telescope_key"],
                "energy_conservation_scale_factor.json",
            )
        )
        mirror_to_camera_energy_scale_factor = ecsf[
            "fitted_energy_scale_factor"
        ]
    else:
        mirror_to_camera_energy_scale_factor = 1.0

    nu_Hz = np.mean(lownoiseblock.input_frequency_start_stop_Hz(tscope["lnb"]))
    wavelength_m = signal.frequency_to_wavelength(nu_Hz)
    num_waves = 7

    region_of_interest_rad = np.arctan(
        (num_waves * wavelength_m) / tscope["mirror"]["focal_length_m"]
    )

    r_100km = 100e3
    A_sphere_100km = psf_utils.area_of_sphere(radius=r_100km)
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
        logger = iaat_logger.LoggerFile(os.path.join(tmp_dir, "log.jsonl"))

        plane_wave_response.make_PlaneWaveResponse(
            out_dir=tmp_dir,
            random_seed=job["id"],
            telescope=tscope,
            site=site,
            timing=timing,
            mirror_to_camera_energy_scale_factor=mirror_to_camera_energy_scale_factor,
            source_config=source_config,
            region_of_interest=job["region_of_interest"],
            region_of_interest_rad=region_of_interest_rad,
            region_of_interest_num_bins=psf_utils.substract_one_when_even(
                num_waves * 6
            ),
            logger=logger,
            save_roi_electric_fields=False,
        )
        response = plane_wave_response.PlaneWaveResponse(tmp_dir)
        response.plot()


def list_response_paths(work_dir, telescope_key, scenario_key):
    response_path_wildcard = os.path.join(
        work_dir, "stars", telescope_key, scenario_key, "*"
    )
    response_paths = glob.glob(response_path_wildcard)
    out_paths = []
    for rpath in response_paths:
        basename = os.path.basename(rpath)
        if can_be_interpreted_as_int(basename):
            if len(basename) == 6:
                out_paths.append(rpath)

    out_paths = sorted(out_paths)
    return out_paths


def can_be_interpreted_as_int(s):
    try:
        v = int(s)
        return True
    except ValueError as err:
        return False
