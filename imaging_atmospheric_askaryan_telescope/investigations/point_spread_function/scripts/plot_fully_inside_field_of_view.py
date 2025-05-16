import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import spherical_coordinates
import binning_utils
import dynamicsizerecarray
import rename_after_writing as rnw


scenario_key = "fully_inside_field_of_view"

parser = argparse.ArgumentParser(
    prog=f"plot_{scenario_key:s}.py",
    description=(f"Plot {scenario_key:s}."),
)
parser.add_argument(
    "psf_dir",
    metavar="PSF_DIR",
    default="point spread function directory",
    type=str,
    help="Path to directory.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

args = parser.parse_args()
psf_dir = args.psf_dir
out_dir = os.path.join(psf_dir, "plots", scenario_key)
os.makedirs(out_dir, exist_ok=True)

config = iaat.investigations.point_spread_function.utils.read_config(psf_dir)

power_ratio_threshold = 0.1

source_key = "1"

snapdtype = [
    ("id", int),
    ("source_azimuth_rad", float),
    ("source_zenith_rad", float),
    ("source_polarization_angle_rad", float),
    ("source_expected_energy_J", float),
    ("signal_to_noise_ratio", float),
    ("energy_conservation_ratio", float),
    ("reconstructed_azimuth_rad", float),
    ("reconstructed_zenith_rad", float),
    ("reconstructed_x_m", float),
    ("reconstructed_y_m", float),
    ("reconstructed_radius_m", float),
    ("reconstructed_polarization_factor", float),
    ("reconstructed_polarization_factor_std", float),
    ("reconstructed_polarization_angle_rad", float),
    ("reconstructed_polarization_angle_std_rad", float),
    ("reconstructed_energy_J", float),
]

for telescope_key in config["stars"]["telescopes"]:
    if "large" in telescope_key:
        continue

    cache_path = os.path.join(out_dir, telescope_key + ".bin")
    if os.path.exists(cache_path):
        continue

    telescope, site, timing = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            config=config, telescope_key=telescope_key
        )
    )
    response_paths = (
        iaat.investigations.point_spread_function.stars.list_response_paths(
            work_dir=psf_dir,
            telescope_key=telescope_key,
            scenario_key=scenario_key,
        )
    )

    airy_radius_m = iaat.telescope.calculate_airy_disk_radius_in_focal_plane(
        telescope=telescope
    )
    airy_angle_rad = airy_radius_m / telescope["mirror"]["focal_length_m"]

    snap = dynamicsizerecarray.DynamicSizeRecarray(dtype=snapdtype)

    for response_path in response_paths:
        response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )

        plane_wave_config = response.source_config["plane_waves"][source_key]

        energy_expected_from_source_J = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
            config=plane_wave_config,
            area_m2=telescope["mirror"]["area_m2"],
        )

        brightest_feed_horn_index = np.argmax(response.energy_feed_horns)
        brightest_feed_horn_position_m = telescope["sensor"][
            "feed_horn_positions_m"
        ][brightest_feed_horn_index]
        bfh_x, bfh_y, _ = brightest_feed_horn_position_m
        bfh_cx = -np.arctan(bfh_x / telescope["mirror"]["focal_length_m"])
        bfh_cy = -np.arctan(bfh_y / telescope["mirror"]["focal_length_m"])
        bfh_az, bfh_zd = spherical_coordinates.cx_cy_to_az_zd(
            cx=bfh_cx, cy=bfh_cy
        )

        src_az = response.source_config["plane_waves"][source_key]["geometry"][
            "azimuth_rad"
        ]
        src_zd = response.source_config["plane_waves"][source_key]["geometry"][
            "zenith_rad"
        ]
        delta_rad = spherical_coordinates.angle_between_az_zd(
            azimuth1_rad=bfh_az,
            zenith1_rad=bfh_zd,
            azimuth2_rad=src_az,
            zenith2_rad=src_zd,
        )

        if delta_rad > 3 * airy_angle_rad:
            print(telescope_key, "delta", np.rad2deg(delta_rad), "deg")

        # print("raio", int(signal_to_noise_ratio))

        xy, w = response.point_cloud_feed_horns_scatter_energy()

        Rmax = 1.2 * telescope["sensor"]["camera"]["outer_radius_m"]
        _w = w / np.percentile(w, 99)
        # gauss_pseudo_2d(xy, x0, y0, sigma)
        guess = [bfh_x, bfh_y, airy_radius_m]
        bounds = (
            [-Rmax, -Rmax, 0.8 * airy_radius_m],
            [Rmax, Rmax, 5 * airy_radius_m],
        )
        try:
            predicted_params, uncert_cov = iaat.utils.curve_fit(
                f=iaat.utils.gauss_pseudo_2d,
                xdata=xy,
                ydata=_w,
                p0=guess,
                bounds=bounds,
            )
        except RuntimeError as err:
            print("Can not fit", response_path)
            continue

        rec_x_m, rec_y_m, rec_radius_m = predicted_params

        rec_cx = -np.arctan(rec_x_m / telescope["mirror"]["focal_length_m"])
        rec_cy = -np.arctan(rec_y_m / telescope["mirror"]["focal_length_m"])

        rec_az_rad, rec_zd_rad = spherical_coordinates.cx_cy_to_az_zd(
            cx=rec_cx, cy=rec_cy
        )

        delta_fit_rad = spherical_coordinates.angle_between_az_zd(
            azimuth1_rad=rec_az_rad,
            zenith1_rad=rec_zd_rad,
            azimuth2_rad=src_az,
            zenith2_rad=src_zd,
        )
        if delta_fit_rad > 3 * airy_angle_rad:
            print(telescope_key, "delta_fit", np.rad2deg(delta_fit_rad), "deg")

        print(rec_x_m, rec_y_m, rec_radius_m)

        feed_horn_signal_mask = iaat.investigations.point_spread_function.plane_wave_response.mask_feed_horns(
            feed_horn_positions_m=telescope["sensor"]["feed_horn_positions_m"],
            containment_radius_m=1.5 * rec_radius_m,
            azimuth_rad=rec_az_rad,
            zenith_rad=rec_zd_rad,
        )
        feed_horn_background_mask = np.logical_not(feed_horn_signal_mask)

        energy_signal = response.energy_feed_horns[feed_horn_signal_mask]
        energy_background = response.energy_feed_horns[
            feed_horn_background_mask
        ]

        total_energy_signal = np.sum(energy_signal)
        mean_energy_density_signal = np.mean(energy_signal)
        median_energy_density_background = np.percentile(energy_background, 50)

        signal_to_noise_ratio = (
            mean_energy_density_signal / median_energy_density_background
        )
        energy_conservation_ratio = (
            total_energy_signal / energy_expected_from_source_J
        )

        # plarization
        # -----------
        rec_factor, rec_pola_rad = (
            iaat.investigations.point_spread_function.polarization_analysis.analyse_linear_polarization(
                electric_fields=response.E_feed_horns,
                channel_mask=feed_horn_signal_mask,
            )
        )

        resultatata = {
            "id": int(os.path.basename(response.path)),
            "source_azimuth_rad": plane_wave_config["geometry"]["azimuth_rad"],
            "source_zenith_rad": plane_wave_config["geometry"]["zenith_rad"],
            "source_polarization_angle_rad": plane_wave_config["geometry"][
                "polarization_angle_rad"
            ],
            "source_expected_energy_J": energy_expected_from_source_J,
            "signal_to_noise_ratio": signal_to_noise_ratio,
            "energy_conservation_ratio": energy_conservation_ratio,
            "reconstructed_azimuth_rad": rec_az_rad,
            "reconstructed_zenith_rad": rec_zd_rad,
            "reconstructed_x_m": rec_x_m,
            "reconstructed_y_m": rec_y_m,
            "reconstructed_radius_m": rec_radius_m,
            "reconstructed_polarization_factor": rec_factor[0],
            "reconstructed_polarization_factor_std": rec_factor[1],
            "reconstructed_polarization_angle_rad": rec_pola_rad[0],
            "reconstructed_polarization_angle_std_rad": rec_pola_rad[1],
            "reconstructed_energy_J": total_energy_signal,
        }
        snap.append_record(resultatata)

    with rnw.open(cache_path, "wb") as f:
        f.write(snap.tobytes())


for telescope_key in config["stars"]["telescopes"]:
    if "large" in telescope_key:
        continue

    telescope, site, timing = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            config=config, telescope_key=telescope_key
        )
    )
    fov = iaat.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    airy_radius_m = iaat.telescope.calculate_airy_disk_radius_in_focal_plane(
        telescope=telescope
    )
    airy_angle_rad = airy_radius_m / telescope["mirror"]["focal_length_m"]

    cache_path = os.path.join(out_dir, telescope_key + ".bin")
    with open(cache_path, "rb") as f:
        snap = np.frombuffer(f.read(), dtype=snapdtype)

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        snap["reconstructed_radius_m"] / airy_radius_m,
        color="black",
        marker="o",
        linewidth=0.0,
        alpha=0.25,
    )
    ax.plot(
        [0, np.rad2deg(fov["field_of_view_half_angle_rad"])],
        [1, 1],
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    ax.set_ylim([0, 3])
    ax.set_xlim([0.0, np.rad2deg(fov["field_of_view_half_angle_rad"])])
    ax.set_xlabel(r"angle off axis / (1$^\circ$)")
    ax.set_ylabel("containmeint radius /\nAiry disk radius")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_spread.jpg"))
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        snap["energy_conservation_ratio"],
        color="black",
        marker="o",
        linewidth=0.0,
        alpha=0.25,
    )
    ax.plot(
        [0, np.rad2deg(fov["field_of_view_half_angle_rad"])],
        [1, 1],
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    ax.set_ylim([0.0, 2.0])
    ax.set_xlim([0.0, np.rad2deg(fov["field_of_view_half_angle_rad"])])
    ax.set_xlabel(r"angle off axis / (1$^\circ$)")
    ax.set_ylabel("energy conservation / 1")
    fig.savefig(
        os.path.join(out_dir, f"{telescope_key:s}_energy_conservation.jpg")
    )
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        snap["reconstructed_zenith_rad"] / snap["source_zenith_rad"],
        color="black",
        marker="o",
        linewidth=0.0,
        alpha=0.25,
    )
    ax.plot(
        [0, np.rad2deg(fov["field_of_view_half_angle_rad"])],
        [1, 1],
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    ax.set_ylim([0.5, 1.5])
    ax.set_xlim([0.0, np.rad2deg(fov["field_of_view_half_angle_rad"])])
    ax.set_xlabel(r"angle off axis / (1$^\circ$)")
    ax.set_ylabel("angle off axis\n reconstructed over true / 1")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_distortion.jpg"))
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        snap["signal_to_noise_ratio"],
        color="black",
        marker="o",
        linewidth=0.0,
        alpha=0.25,
    )
    ax.set_ylim([1e0, 1e3])
    ax.semilogy()
    ax.set_xlim([0.0, np.rad2deg(fov["field_of_view_half_angle_rad"])])
    ax.set_xlabel(r"angle off axis / (1$^\circ$)")
    ax.set_ylabel("signal to noise / 1")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_snr.jpg"))
    sebplt.close(fig)

    pola_diff_rad = np.mod(
        snap["source_polarization_angle_rad"], np.pi
    ) - np.mod(snap["reconstructed_polarization_angle_rad"], np.pi)
    pola_diff_rad = np.mod(pola_diff_rad, np.pi)
    pola_order_mask = pola_diff_rad > np.pi / 2
    pola_diff_rad[pola_order_mask] = pola_diff_rad[pola_order_mask] - np.pi

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        np.rad2deg(pola_diff_rad),
        color="black",
        marker="o",
        linewidth=0.0,
        alpha=0.05,
    )
    ax.set_ylim([-100, 100])
    ax.set_xlim([0.0, np.rad2deg(fov["field_of_view_half_angle_rad"])])
    ax.set_xlabel(r"angle off axis / (1$^\circ$)")
    ax.set_ylabel(
        "true - reconstructed\npolarization angle / " + r"(1$^\circ$)"
    )
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_polarization.jpg"))
    sebplt.close(fig)
