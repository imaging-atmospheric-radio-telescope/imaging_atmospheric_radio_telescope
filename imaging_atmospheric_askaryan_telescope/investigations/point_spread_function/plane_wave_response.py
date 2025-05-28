from . import utils as psf_utils
from . import plot as psf_plot
from ... import calibration_source
from ... import production
from ... import time_series
from ... import electric_fields
from ... import signal
from ... import camera

import rename_after_writing as rnw
import spherical_coordinates
import os
import numpy as np
import json_utils
import shutil
import glob


def make_PlaneWaveResponse(
    out_dir,
    random_seed,
    telescope,
    site,
    timing,
    source_config,
    region_of_interest=True,
    region_of_interest_rad=np.deg2rad(0.5),
    region_of_interest_num_bins=42,
    save_feed_horns_scatter_electric_fields=False,
    save_roi_electric_fields=False,
    logger=None,
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
        save_feed_horns_scatter_electric_fields=save_feed_horns_scatter_electric_fields,
        logger=logger,
    )

    with rnw.open(os.path.join(camera_dir, "sensor.json"), "wt") as f:
        f.write(json_utils.dumps(telescope["sensor"], indent=4))

    if region_of_interest:
        roi_dir = os.path.join(out_dir, "region_of_interest")

        for key in source_config["plane_waves"]:
            roi_key_dir = os.path.join(roi_dir, key)

            plane_wave_config = source_config["plane_waves"][key]

            telescope_region_of_interest = psf_utils.make_telescope_like_other_but_with_region_of_interest_camera(
                source_azimuth_rad=plane_wave_config["geometry"][
                    "azimuth_rad"
                ],
                source_zenith_rad=plane_wave_config["geometry"]["zenith_rad"],
                region_of_interest_rad=region_of_interest_rad,
                num_bins=region_of_interest_num_bins,
                other_telescope=telescope,
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
                save_feed_horns_electric_fields=save_roi_electric_fields,
                logger=logger,
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
        if not hasattr(self, "_region_of_interest_keys"):
            _p = glob.glob(os.path.join(self.path, "region_of_interest", "*"))
            self._region_of_interest_keys = [os.path.basename(p) for p in _p]
        return self._region_of_interest_keys

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
    def E_feed_horns(self):
        if not hasattr(self, "_E_feed_horns"):
            self._E_feed_horns = time_series.read(
                os.path.join(
                    self.path, "camera", "feed_horns", "electric_fields.tar"
                )
            )
        return self._E_feed_horns

    @property
    def E_feed_horns_scatter(self):
        if not hasattr(self, "_E_feed_horns_scatter"):
            self._E_feed_horns_scatter = time_series.read(
                os.path.join(
                    self.path,
                    "camera",
                    "feed_horns",
                    "scatter.electric_fields.tar",
                )
            )
        return self._E_feed_horns_scatter

    @property
    def energy_feed_horns_scatter(self):
        _path = os.path.join(
            self.path,
            "camera",
            "feed_horns",
            "scatter.energy.npy",
        )
        with open(_path, "rb") as f:
            ene = np.load(f)
        return ene

    def point_cloud_feed_horns_scatter_energy(self):
        sc_pos = camera.get_camera_feed_horn_scatter_centers(self.sensor)
        sc_w = self.energy_feed_horns_scatter
        return sc_pos[:, 0:2], sc_w

    def plot_energy_feed_horns_scatter(self, path):
        energy_feed_horns_scatter_eV = (
            self.energy_feed_horns_scatter / signal.ELECTRON_VOLT_J
        )
        psf_plot.plot_feed_horn_scatter_centers(
            camera=self.sensor,
            energy_feed_horns_scatter_eV=energy_feed_horns_scatter_eV,
            path=path,
        )

    def plot_energy_feed_horns(self, path):
        energy_feed_horns_eV = self.energy_feed_horns / signal.ELECTRON_VOLT_J
        psf_plot.plot_camera(
            camera=self.sensor,
            energy_feed_horns_eV=energy_feed_horns_eV,
            path=path,
        )

    def plot(self):
        _fig_path = os.path.join(self.path, "feed_horns_scatter.jpg")
        if not os.path.exists(_fig_path):
            self.plot_energy_feed_horns_scatter(path=_fig_path)

        _fig_path = os.path.join(self.path, "feed_horns.jpg")
        if not os.path.exists(_fig_path):
            self.plot_energy_feed_horns(path=_fig_path)

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
    def energy_feed_horns(self):
        _path = os.path.join(
            self.path,
            "camera",
            "feed_horns",
            "energy.npy",
        )
        with open(_path, "rb") as f:
            Ene_J = np.load(f)
        return Ene_J

    def energy_roi(self, key):
        sensor_roi_key = self.sensor_roi(key)

        _path = os.path.join(
            self.path,
            "region_of_interest",
            key,
            "feed_horns",
            "energy.npy",
        )
        with open(_path, "rb") as f:
            Ene_roi_J = np.load(f)

        x_bin_edges = sensor_roi_key["region_of_interest"]["x_bin_edges_m"]
        y_bin_edges = sensor_roi_key["region_of_interest"]["y_bin_edges_m"]

        Ene_roi_J = Ene_roi_J.reshape(
            (
                len(x_bin_edges) - 1,
                len(y_bin_edges) - 1,
            )
        )
        return x_bin_edges, y_bin_edges, Ene_roi_J


def mask_feed_horns(
    feed_horn_positions_m,
    containment_radius_m,
    azimuth_rad,
    zenith_rad,
):
    cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(
        azimuth_rad=azimuth_rad,
        zenith_rad=zenith_rad,
    )
    p_xyz = np.array([-cx, -cy, cz])
    feed_horn_z_m = np.mean(feed_horn_positions_m[:, 2])
    scale_factor = feed_horn_z_m / cz
    expected_spot_in_camera_screen_m = p_xyz * scale_factor

    return mask_feed_horns_x_y(
        feed_horn_positions_m=feed_horn_positions_m,
        containment_radius_m=containment_radius_m,
        x_m=expected_spot_in_camera_screen_m[0],
        y_m=expected_spot_in_camera_screen_m[1],
    )


def mask_feed_horns_x_y(
    feed_horn_positions_m,
    containment_radius_m,
    x_m,
    y_m,
):
    feed_horn_z_m = np.mean(feed_horn_positions_m[:, 2])
    expected_spot_in_camera_screen_m = [x_m, y_m, feed_horn_z_m]

    mask = np.zeros(feed_horn_positions_m.shape[0], dtype=bool)
    for i in range(feed_horn_positions_m.shape[0]):
        delta_m = np.linalg.norm(
            feed_horn_positions_m[i] - expected_spot_in_camera_screen_m
        )
        if delta_m <= containment_radius_m:
            mask[i] = True
        else:
            mask[i] = False
    return mask
