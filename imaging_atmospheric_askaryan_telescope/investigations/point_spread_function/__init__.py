from . import plot

from ... import telescope
from ... import telescopes
from ... import sites
from ... import signal
from ... import production
from ... import time_series
from ... import electric_fields
from ... import utils
from ... import lownoiseblock
from ... import timing_and_sampling
from ... import calibration_source

import spherical_coordinates
import numpy as np
import os
import json_utils
import rename_after_writing as rnw
import shutil
import glob
import scipy
from astropy.convolution.kernels import Gaussian2DKernel


def init(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # telescopes
    telescopes_dir = os.path.join(config_dir, "telescopes")
    os.makedirs(telescopes_dir, exist_ok=True)

    for key in ["crome", "large_size_telescope"]:
        telescope_config = telescopes.init(key)
        with rnw.open(
            os.path.join(telescopes_dir, f"{key:s}.json"), "wt"
        ) as f:
            f.write(json_utils.dumps(telescope_config, indent=4))

    with rnw.open(os.path.join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(sites.init("karlsruhe"), indent=4))

    timing = {
        "oversampling": 6,
        "time_window_duration_s": 3.5e-08,
        "readout_sampling_rate_per_s": 250e6,
    }
    with rnw.open(
        os.path.join(config_dir, "timing_and_sampling.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(timing, indent=4))

    stars = {
        "telescopes": ["crome", "large_size_telescope"],
        "random_seed": 1,
        "num": 8,
        "power_density_start_W_per_m2": 1e-12,
        "power_density_stop_W_per_m2": 3e-12,
    }
    with rnw.open(os.path.join(config_dir, "stars.json"), "wt") as f:
        f.write(json_utils.dumps(stars, indent=4))


def run(work_dir, pool=None, logger=None):
    pool = _serial_pool_if_None(pool)
    logger = _stdout_logger_if_None(logger)
    config = _read_config(work_dir)

    logger.debug("make jobs for 'stars' ...")
    star_jobs = _star_make_jobs(work_dir=work_dir, config=config)
    star_jobs = _star_drop_finished_jobs(work_dir=work_dir, jobs=star_jobs)
    logger.debug(f"f{len(star_jobs):d} jobs are missing and need to be run.")

    logger.debug("run jobs for 'stars' ...")
    pool.map(_star_run_job, star_jobs)


def _star_make_jobs(work_dir, config):
    prng = np.random.Generator(np.random.PCG64(4))

    jobs = []
    for telescope_key in config["stars"]["telescopes"]:
        tscope, _, _ = _make_telescope_timing_and_site(
            config=config, telescope_key=telescope_key
        )
        nu_start_Hz, nu_stop_Hz = lownoiseblock.input_frequency_start_stop_Hz(
            lnb=tscope["lnb"]
        )

        camera_screen_min_radius_m = (
            tscope["sensor"]["camera"]["outer_radius_m"]
            - tscope["sensor"]["camera"]["feed_horn_inner_radius_m"]
        )
        max_angle_off_axis_rad = np.arctan(
            camera_screen_min_radius_m / tscope["mirror"]["focal_length_m"]
        )

        for i in range(config["stars"]["num"]):
            job = {}
            job["id"] = i
            job["telescope_key"] = telescope_key
            job["work_dir"] = work_dir
            job["random_seed"] = config["stars"]["random_seed"] + i
            job["path"] = os.path.join(
                work_dir, "stars", telescope_key, f"{i:06d}"
            )
            az_rad, zd_rad = (
                spherical_coordinates.random.uniform_az_zd_in_cone(
                    prng=prng,
                    azimuth_rad=0.0,
                    zenith_rad=0.0,
                    min_half_angle_rad=0.0,
                    max_half_angle_rad=max_angle_off_axis_rad,
                )
            )
            job["source_azimuth_rad"] = az_rad
            job["source_zenith_rad"] = zd_rad
            job["source_polarization_angle_rad"] = prng.uniform(
                low=0.0, high=2.0 * np.pi
            )
            job["power_density_W_per_m2"] = prng.uniform(
                low=config["stars"]["power_density_start_W_per_m2"],
                high=config["stars"]["power_density_stop_W_per_m2"],
            )
            job["frequency_Hz"] = prng.uniform(
                low=nu_start_Hz,
                high=nu_stop_Hz,
            )
            jobs.append(job)
    return jobs


def _star_drop_finished_jobs(work_dir, jobs):
    out = []
    for job in jobs:
        if not os.path.exists(os.path.join(job["path"])):
            out.append(job)
    return out


def _star_run_job(job):
    config = _read_config(job["work_dir"])

    tscope, timing, site = _make_telescope_timing_and_site(
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
        make_PlaneWaveResponse(
            out_dir=tmp_dir,
            random_seed=job["random_seed"],
            telescope=tscope,
            site=site,
            timing=timing,
            source_config=source_config,
            region_of_interest_rad=region_of_interest_rad,
            region_of_interest_num_bins=substract_one_when_even(
                num_waves * timing["oversampling"]
            ),
        )


def substract_one_when_even(x):
    if np.mod(x, 2) > 0:
        return x - 1
    else:
        return x


def _make_telescope_timing_and_site(config, telescope_key):
    telescope_config = config["telescopes"][telescope_key]

    _lnb = lownoiseblock.init(key=telescope_config["lnb_key"])
    _mirror = telescope.make_mirror(**telescope_config["mirror"])
    _sensor = telescope.make_sensor(**telescope_config["sensor"])

    tscope = telescope.make_telescope(
        sensor=_sensor,
        mirror=_mirror,
        lnb=_lnb,
        speed_of_light_m_per_s=signal.SPEED_OF_LIGHT_M_PER_S,
    )
    timing = timing_and_sampling.make_timing_from_lnb(
        lnb=tscope["lnb"],
        **config["timing_and_sampling"],
    )
    return tscope, timing, config["site"]


def make_telescope_like_other_but_with_region_of_interest_camera(
    source_azimuth_rad,
    source_zenith_rad,
    other_telescope,
    region_of_interest_rad,
    num_bins,
):
    roi_rad = region_of_interest_rad
    f = other_telescope["mirror"]["focal_length_m"]
    px_center_rad, py_center_rad = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=source_azimuth_rad,
        zenith_rad=source_zenith_rad,
    )
    cx_center_rad = -px_center_rad
    cy_center_rad = -py_center_rad

    x_bin_edges_m = f * np.linspace(
        cx_center_rad - roi_rad / 2,
        cx_center_rad + roi_rad / 2,
        num_bins + 1,
    )
    y_bin_edges_m = f * np.linspace(
        cy_center_rad - roi_rad / 2,
        cy_center_rad + roi_rad / 2,
        num_bins + 1,
    )

    sensor_roi = telescope.make_sensor_in_region_of_interest(
        x_bin_edges_m=x_bin_edges_m,
        y_bin_edges_m=y_bin_edges_m,
        sensor_distance_m=other_telescope["sensor"]["sensor_distance_m"],
        feed_horn_transmission=1.0,
    )

    return telescope.make_telescope_like_other_but_different_sensor(
        telescope=other_telescope,
        sensor=sensor_roi,
    )


def make_PlaneWaveResponse(
    out_dir,
    random_seed,
    telescope,
    site,
    timing,
    source_config,
    region_of_interest_rad=np.deg2rad(0.5),
    region_of_interest_num_bins=42,
):
    os.makedirs(out_dir, exist_ok=True)
    camera_dir = os.path.join(out_dir, "camera")

    with rnw.open(os.path.join(out_dir, "source_config.json"), "wt") as f:
        f.write(json_utils.dumps(source_config, indent=4))

    production.simulate_telescope_response(
        out_dir=camera_dir,
        source_config=source_config,
        site=site,
        telescope=telescope,
        timing=timing,
        thermal_noise_random_seed=random_seed + 1,
        readout_random_seed=random_seed + 2,
        camera_lnb_random_seed=random_seed + 3,
        stop_after_section="feed_horns",
    )

    with rnw.open(os.path.join(camera_dir, "sensor.json"), "wt") as f:
        f.write(json_utils.dumps(telescope["sensor"], indent=4))

    roi_dir = os.path.join(out_dir, "region_of_interest")

    for key in source_config["plane_waves"]:
        roi_key_dir = os.path.join(roi_dir, key)

        plane_wave_config = source_config["plane_waves"][key]

        telescope_region_of_interest = (
            make_telescope_like_other_but_with_region_of_interest_camera(
                source_azimuth_rad=plane_wave_config["geometry"][
                    "azimuth_rad"
                ],
                source_zenith_rad=plane_wave_config["geometry"]["zenith_rad"],
                region_of_interest_rad=region_of_interest_rad,
                num_bins=region_of_interest_num_bins,
                other_telescope=telescope,
            )
        )

        os.makedirs(roi_key_dir, exist_ok=True)
        shutil.copytree(
            src=os.path.join(camera_dir, "mirror"),
            dst=os.path.join(roi_key_dir, "mirror"),
        )

        production.simulate_telescope_response(
            out_dir=roi_key_dir,
            source_config=source_config,
            site=site,
            telescope=telescope_region_of_interest,
            timing=timing,
            thermal_noise_random_seed=random_seed + 1,
            readout_random_seed=random_seed + 2,
            camera_lnb_random_seed=random_seed + 3,
            stop_after_section="feed_horns",
        )

        with rnw.open(os.path.join(roi_key_dir, "sensor.json"), "wt") as f:
            f.write(
                json_utils.dumps(
                    telescope_region_of_interest["sensor"], indent=4
                )
            )

        shutil.rmtree(os.path.join(roi_key_dir, "mirror"))


class PlaneWaveResponse:
    def __init__(self, path):
        self.path = path
        self._E_roi = {}
        self._sensor_roi = {}

    def __repr__(self):
        smodule = self.__module__
        sname = self.__class__.__name__
        return f"{smodule:s}.{sname:s}(path='{self.path:s}')"

    @property
    def source_config(self):
        if not hasattr(self, "_source_config"):
            with open(
                os.path.join(self.path, "source_config.json"),
                "rt",
            ) as f:
                self._source_config = json_utils.loads(f.read())
        return self._source_config

    @property
    def region_of_interest_keys(self):
        return list(self.source_config["plane_waves"].keys())

    @property
    def E_mirror(self):
        if not hasattr(self, "_E_mirror"):
            self._E_mirror = time_series.read(
                os.path.join(
                    self.path, "camera", "mirror", "electric_fields.tar"
                )
            )
        return self._E_mirror

    @property
    def E_camera(self):
        if not hasattr(self, "_E_camera"):
            self._E_camera = time_series.read(
                os.path.join(
                    self.path, "camera", "feed_horns", "electric_fields.tar"
                )
            )
        return self._E_camera

    def E_roi(self, key):
        if key not in self._E_roi:
            self._E_roi[key] = time_series.read(
                os.path.join(
                    self.path,
                    "region_of_interest",
                    key,
                    "feed_horns",
                    "electric_fields.tar",
                )
            )
        return self._E_roi[key]

    def sensor_roi(self, key):
        if key not in self._sensor_roi:
            with open(
                os.path.join(
                    self.path, "region_of_interest", key, "sensor.json"
                ),
                "rt",
            ) as f:
                self._sensor_roi[key] = json_utils.loads(f.read())
        return self._sensor_roi[key]

    @property
    def sensor(self):
        if not hasattr(self, "_sensor"):
            with open(
                os.path.join(self.path, "camera", "sensor.json"),
                "rt",
            ) as f:
                self._sensor = json_utils.loads(f.read())
        return self._sensor

    @property
    def Image_energy(self):
        Ene_J = electric_fields.integrate_power_over_time(
            electric_fields=self.E_camera,
            channel_effective_area_m2=self.sensor["feed_horn_area_m2"],
        )
        return Ene_J

    def Image_energy_roi(self, key):
        E_roi_key = self.E_roi(key)
        sensor_roi_key = self.sensor_roi(key)
        Ene_roi_J = electric_fields.integrate_power_over_time(
            electric_fields=E_roi_key,
            channel_effective_area_m2=sensor_roi_key["feed_horn_area_m2"],
        )
        x_bin_edges = sensor_roi_key["region_of_interest"]["x_bin_edges_m"]
        y_bin_edges = sensor_roi_key["region_of_interest"]["y_bin_edges_m"]

        Ene_roi_J = Ene_roi_J.reshape(
            (
                len(x_bin_edges) - 1,
                len(y_bin_edges) - 1,
            )
        )
        return x_bin_edges, y_bin_edges, Ene_roi_J


def make_2d_gaussian_convolution_kernel(width, std=0.2):
    return Gaussian2DKernel(
        x_stddev=width * std,
        x_size=width,
        y_size=width,
    ).array


def oversample_image_twice(image):
    oversampling = 2
    out = np.zeros(
        shape=(
            image.shape[0] * oversampling,
            image.shape[1] * oversampling,
        )
    )
    for nx in range(out.shape[0]):
        for ny in range(out.shape[1]):
            out[nx, ny] = image[nx // 2, ny // 2]
    return out


def _analyse_image(image, containment_quantile=0.8):
    num_bins_quantile = find_quantile_bins(image, q=containment_quantile)

    kernel_width = int(np.round(np.sqrt(num_bins_quantile)))
    kernel_width = np.max([3, kernel_width])
    o_kernel_width = 2 * kernel_width

    o_image = oversample_image_twice(image)
    o_kernel = make_2d_gaussian_convolution_kernel(width=o_kernel_width)
    o_smooth_image = scipy.signal.convolve2d(
        in1=o_image,
        in2=o_kernel,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
    )

    o_argmax = utils.argmaxNd(o_smooth_image)

    return {
        "num_bins_quantile": num_bins_quantile,
        "argmax_x_bin": o_argmax[0] / 2,
        "argmax_y_bin": o_argmax[1] / 2,
    }


def analyse_image(
    x_bin_edges_m, y_bin_edges_m, image, containment_quantile=0.8
):
    ana = _analyse_image(
        image=image, containment_quantile=containment_quantile
    )
    bx = x_bin_edges_m
    by = y_bin_edges_m

    ccx = np.interp(x=ana["argmax_x_bin"], xp=np.arange(0, len(bx)), fp=bx)
    ccy = np.interp(x=ana["argmax_y_bin"], xp=np.arange(0, len(by)), fp=by)
    x_bin_width = np.mean(np.gradient(bx))
    y_bin_width = np.mean(np.gradient(by))
    assert 0.9 < x_bin_width / y_bin_width < 1.1

    ana["argmax_x_m"] = ccx
    ana["argmax_y_m"] = ccy
    ana["area_quantile_m2"] = ana["num_bins_quantile"] * x_bin_width**2
    ana["radius_quantile_m"] = np.sqrt(ana["area_quantile_m2"] / np.pi)

    return ana


def find_quantile_bins(x, q):
    f = np.flip(np.sort(x.flatten()))
    total = np.sum(f)
    fraction = total * q
    cumsum_f = np.cumsum(f)
    idx = np.argmin(np.abs(cumsum_f - fraction))
    return idx


def _serial_pool_if_None(pool):
    return utils.SerialPool() if pool is None else pool


def _read_config(work_dir):
    config = json_utils.tree.read(os.path.join(work_dir, "config"))
    config = utils.strip_dict(config, "comment")
    return config
