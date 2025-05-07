from . import plot

from ... import telescope
from ... import production
from ... import time_series
from ... import electric_fields
from ... import utils

import spherical_coordinates
import numpy as np
import os
import json_utils
import rename_after_writing as rnw
import shutil
import glob
import scipy
from astropy.convolution.kernels import Gaussian2DKernel


def init_work_dir(work_dir, telescope_key):
    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    scatter = {}
    with rnw.open(os.path.join(config_dir, "scatter"), "wt") as f:
        f.write(json_utils.dumps(scatter, indent=4))


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


def analyse_image(image, containment_quantile=0.8):
    Q = np.quantile(image, q=containment_quantile)
    num_bins_quantile = np.sum(image >= Q)

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
