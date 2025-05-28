import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import spherical_coordinates
import binning_utils
import dynamicsizerecarray
import json_utils
import rename_after_writing as rnw
import pandas


scenario_key = "point_spread_function"

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

source_key = "1"


def report_add_source(report, telescope, plane_wave_config):
    report["source_azimuth_rad"] = plane_wave_config["geometry"]["azimuth_rad"]
    report["source_zenith_rad"] = plane_wave_config["geometry"]["zenith_rad"]
    report["source_polarization_angle_rad"] = plane_wave_config["geometry"][
        "polarization_angle_rad"
    ]
    report["source_expected_energy_J"] = (
        iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
            config=plane_wave_config,
            area_m2=telescope["mirror"]["area_m2"],
        )
    )
    return report


def report_add_id(report, response_path):
    report["id"] = int(os.path.basename(response_path))
    return report


def report_add_roi_analysis(report, telescope, roi_analysis):
    report["roi_area_80p_m2"] = roi_analysis["area_quantile_m2"]
    report["roi_x_m"] = roi_analysis["argmax_x_m"]
    report["roi_y_m"] = roi_analysis["argmax_y_m"]
    az, zd = iaat.utils.sky_and_screen.screen_x_y_to_sky_az_zd(
        x_m=report["roi_x_m"],
        y_m=report["roi_y_m"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    report["roi_azimuth_rad"] = az
    report["roi_zenith_rad"] = zd
    report["roi_solid_angle_80p_sr"] = (
        iaat.utils.sky_and_screen.screen_area_to_sky_solid_angle(
            a_m2=report["roi_area_80p_m2"],
            focal_length_m=telescope["mirror"]["focal_length_m"],
        )
    )
    return report


def make_feed_horns_signal_mask(feed_horn_positions_m, x_m, y_m, r_m):
    mask = np.zeros(feed_horn_positions_m.shape[0], dtype=bool)
    for i in range(feed_horn_positions_m.shape[0]):
        fx, fy, _ = feed_horn_positions_m[i]
        d = np.hypot((fx - x_m), (fy - y_m))
        if d <= r_m:
            mask[i] = True
    return mask


def ax_add_fov_marker(ax, x):
    ax.axvline(x, linestyle="--", color="black", alpha=0.25)


PSF_QUANTILE = 0.8

for telescope_key in config["stars"]["telescopes"]:

    cache_path = os.path.join(out_dir, telescope_key + ".jsonl")
    if os.path.exists(cache_path):
        continue

    telescope, site, timing = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            work_dir=psf_dir, config=config, telescope_key=telescope_key
        )
    )
    response_paths = (
        iaat.investigations.point_spread_function.stars.list_response_paths(
            work_dir=psf_dir,
            telescope_key=telescope_key,
            scenario_key="fully_inside_field_of_view",
        )
    )
    response_paths += (
        iaat.investigations.point_spread_function.stars.list_response_paths(
            work_dir=psf_dir,
            telescope_key=telescope_key,
            scenario_key="on_edge_of_field_of_view",
        )
    )

    airy_radius_m = iaat.telescope.calculate_airy_disk_radius_in_focal_plane(
        telescope=telescope
    )
    airy_angle_rad = airy_radius_m / telescope["mirror"]["focal_length_m"]

    reports = []
    for response_path in response_paths:
        response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )

        plane_wave_config = response.source_config["plane_waves"][source_key]

        report = {}
        report = report_add_id(report=report, response_path=response_path)
        report = report_add_source(
            report=report,
            telescope=telescope,
            plane_wave_config=plane_wave_config,
        )

        x_bin_edges_m, y_bin_edges_m, Ene_roi_J = response.energy_roi(
            source_key
        )
        roi_analysis = iaat.investigations.point_spread_function.power_image_analysis.analyse_image(
            x_bin_edges_m=x_bin_edges_m,
            y_bin_edges_m=y_bin_edges_m,
            image=Ene_roi_J,
            containment_quantile=0.8,
        )
        report["roi_r80_m"] = (
            iaat.investigations.point_spread_function.power_image_analysis.encircle_containment(
                x_bin_edges_m=x_bin_edges_m,
                y_bin_edges_m=y_bin_edges_m,
                image=Ene_roi_J,
                x_m=roi_analysis["argmax_x_m"],
                y_m=roi_analysis["argmax_y_m"],
                quantile=PSF_QUANTILE,
            )
        )

        """
        fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 1.5})
        ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
        ax_cmap = sebplt.add_axes(fig=fig, span=[0.83, 0.15, 0.025, 0.65])
        norm = sebplt.matplotlib.colors.PowerNorm(
            vmin=1e-3 * np.max(Ene_roi_J),
            vmax=np.max(Ene_roi_J),
            gamma=1 / 2.0,
        )
        im = ax.pcolormesh(
            x_bin_edges_m,
            y_bin_edges_m,
            Ene_roi_J.T,
            cmap="Blues",
            norm=norm,
        )
        ax.plot(roi_analysis["argmax_x_m"], roi_analysis["argmax_y_m"], marker="o", color="red")
        sebplt.ax_add_circle(
            ax=ax,
            x=roi_analysis["argmax_x_m"],
            y=roi_analysis["argmax_y_m"],
            r=roi_analysis["radius_quantile_m"],
            linestyle="--",
            color="red",
        )
        sebplt.ax_add_circle(
            ax=ax,
            x=roi_analysis["argmax_x_m"],
            y=roi_analysis["argmax_y_m"],
            r=report["roi_r80_m"],
            linestyle="--",
            color="green",
        )
        ax.set_xlim([min(x_bin_edges_m), max(x_bin_edges_m)])
        ax.set_ylim([min(y_bin_edges_m), max(y_bin_edges_m)])
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_aspect("equal")
        sebplt.plt.colorbar(im, cax=ax_cmap)
        ax_cmap.set_ylabel(r"Energy / eV")

        fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_{report['id']:06d}.jpg"))
        sebplt.close(fig)
        """

        feed_horns_signal_mask = make_feed_horns_signal_mask(
            feed_horn_positions_m=telescope["sensor"]["feed_horn_positions_m"],
            x_m=roi_analysis["argmax_x_m"],
            y_m=roi_analysis["argmax_y_m"],
            r_m=2 * airy_radius_m,
        )
        feed_horns_background_mask = np.logical_not(feed_horns_signal_mask)

        energy_signal_J = response.energy_feed_horns[feed_horns_signal_mask]
        energy_background_J = response.energy_feed_horns[
            feed_horns_background_mask
        ]

        total_energy_signal_J = np.sum(energy_signal_J)
        mean_energy_signal_J = np.mean(energy_signal_J)
        median_energy_background_J = np.percentile(energy_background_J, 50)

        signal_to_noise_ratio = (
            mean_energy_signal_J / median_energy_background_J
        )
        energy_conservation_ratio = (
            total_energy_signal_J / report["source_expected_energy_J"]
        )

        report = report_add_roi_analysis(
            report=report, telescope=telescope, roi_analysis=roi_analysis
        )

        report["feed_horn_total_energy_signal_J"] = total_energy_signal_J
        report["feed_horn_median_energy_background_J"] = (
            median_energy_background_J
        )
        report["feed_horn_signal_to_noise_ratio"] = signal_to_noise_ratio
        report["feed_horn_energy_conservation_ratio"] = (
            energy_conservation_ratio
        )
        reports.append(report)

    json_utils.lines.write(cache_path, reports)


def read_reports(path):
    reports = json_utils.lines.read(path)
    df = pandas.DataFrame.from_records(reports)
    return df.to_records(index=False)


def percentile_spread(x, p):
    p_half = p / 2
    x_start = np.percentile(x, 50 - p_half)
    x_stop = np.percentile(x, 50 + p_half)
    return x_stop - x_start


def ax_square_format(ax):
    xt_deg2 = np.array(ax.get_xticks())
    xticklabels = []
    for xx_deg2 in xt_deg2:
        if xx_deg2 >= 0.0:
            xtl = r"{:.2f}".format(np.sqrt(xx_deg2)) + r"$^{2}$"
        else:
            xtl = r""
        xticklabels.append(xtl)

    ax.set_xticks(ax.get_xticks())  # to get rid of UserWarning
    ax.set_xticklabels(xticklabels)


def ax_blank_format(ax):
    xt = np.array(ax.get_xticks())
    xticklabels = ["" for xx in xt]
    ax.set_xticks(ax.get_xticks())  # to get rid of UserWarning
    ax.set_xticklabels(xticklabels)


def sci_decimals(x):
    if x > 1.0:
        return 0
    else:
        return int(np.ceil(np.abs(np.log10(x))))


def sci_format(decimals):
    return "{:" + f".{decimals:d}f" + "}"


def sci_uncertainty(x, dx):
    decimals = sci_decimals(dx)
    sx = sci_format(decimals).format(x)
    sdx = sci_format(decimals).format(dx)
    return sx + "\\pm" + sdx


def ax_add_uncertain_bins(ax, x_bin_edges, y, y_std, weights, **kwargs):
    _num_bins = len(x_bin_edges) - 1

    for ibin in range(_num_bins):
        xx_start = x_bin_edges[ibin]
        xx_stop = x_bin_edges[ibin + 1]

        WWW = 10
        for i in range(WWW):
            www = (i + 1) / WWW

            yy_start = y[ibin] - www * 0.5 * y_std[ibin]
            yy_stop = y[ibin] + www * 0.5 * y_std[ibin]
            ax.fill(
                [xx_start, xx_stop, xx_stop, xx_start],
                [yy_start, yy_start, yy_stop, yy_stop],
                alpha=weights[ibin] / WWW,
                linewidth=0.0,
                **kwargs,
            )


SHOW_FIT = False
XLABEL_OFF_AXIS_DEG2 = (
    r"(angle off the mirror's optical axis)$^{2}\,/\,(1^{\circ{}})^{2}$"
)

for telescope_key in config["stars"]["telescopes"]:

    telescope, site, timing = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            work_dir=psf_dir, config=config, telescope_key=telescope_key
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
    airy_area_m2 = np.pi * airy_radius_m**2

    cache_path = os.path.join(out_dir, telescope_key + ".jsonl")
    snap = read_reports(cache_path)

    psf_area_m2 = np.pi * snap["roi_r80_m"] ** 2
    psf_off_deg = np.rad2deg(snap["source_zenith_rad"])

    off_num_bins = int(np.sqrt(0.5 * len(psf_off_deg)))
    oa_bin = binning_utils.Binning(
        bin_edges=np.linspace(
            0.0,
            np.rad2deg(fov["field_of_view_half_angle_rad"]) ** 2,
            off_num_bins + 1,
        )
        ** 0.5
    )

    FOV_HA_DEG = np.rad2deg(fov["field_of_view_half_angle_rad"])
    OFF_STOP_DEG = np.sqrt(np.ceil(FOV_HA_DEG**2))

    # AREAL SPREAD
    # ============
    h1_area_p50 = np.zeros(oa_bin["num"])
    h1_area_s68 = np.zeros(oa_bin["num"])
    h1_cnt = np.zeros(oa_bin["num"])

    for isi in range(oa_bin["num"]):
        off2_start = oa_bin["edges"][isi]
        off2_stop = oa_bin["edges"][isi + 1]
        off2_mask = np.logical_and(
            psf_off_deg >= off2_start, psf_off_deg < off2_stop
        )
        cnt = np.sum(off2_mask)
        h1_cnt[isi] = cnt
        if cnt > 0:
            h1_area_p50[isi] = np.percentile(psf_area_m2[off2_mask], 50)
            h1_area_s68[isi] = percentile_spread(psf_area_m2[off2_mask], 68)
        else:
            h1_area_p50[isi] = float("nan")
            h1_area_s68[isi] = float("nan")

    AREA_SCALE = 1 / airy_area_m2

    relative_cnt = np.sqrt(h1_cnt)
    relative_cnt = relative_cnt / np.max(relative_cnt)
    relative_cnt[relative_cnt > 1] = 1.0

    # FIT
    try:
        fit_coef, fit_cov = np.polyfit(
            x=oa_bin["centers"], y=h1_area_p50 * AREA_SCALE, deg=1, cov=True
        )
        fit_coef_std = np.sqrt(np.diag(fit_cov))
        psf_fit_airy_per_deg = np.poly1d(fit_coef)

        with rnw.open(
            os.path.join(out_dir, f"{telescope_key:s}_spread.txt"), "wt"
        ) as f:
            f.write("Area/Airy = ")
            f.write("(")
            f.write(sci_uncertainty(fit_coef[0], fit_coef_std[0]))
            f.write(")")
            f.write(" angle/deg + ")
            f.write("(")
            f.write(sci_uncertainty(fit_coef[1], fit_coef_std[1]))
            f.write(")")
        HAVE_FIT = True
    except ValueError:
        HAVE_FIT = False

    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.05, 0.75, 0.9])

    ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=oa_bin["edges"] ** 2,
        y=h1_area_p50 * AREA_SCALE,
        y_std=h1_area_s68 * AREA_SCALE,
        weights=relative_cnt,
        color="black",
    )

    if SHOW_FIT and HAVE_FIT:
        _xxx = np.linspace(oa_bin["start"], oa_bin["stop"], 201)
        ax.plot(_xxx**2, psf_fit_airy_per_deg(_xxx), "-r")

    ax_add_fov_marker(ax, FOV_HA_DEG**2)
    ylim = [0.0, 1.25 * np.nanmax(h1_area_p50) * AREA_SCALE]
    ax.set_ylim(ylim)
    ax.set_xlim([0.0, OFF_STOP_DEG**2])
    ax_blank_format(ax=ax)
    ax.set_ylabel("area containing 80% /\n" + "Airy disk")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_spread.jpg"))
    sebplt.close(fig)

    # COUNTS
    # ======
    fig = sebplt.figure(style={"rows": 600, "cols": 1920, "fontsize": 2.0})
    ax_cnt = sebplt.add_axes(fig=fig, span=[0.2, 0.4, 0.75, 0.55])
    ax_cnt.set_xlabel(XLABEL_OFF_AXIS_DEG2)
    ax_cnt.set_ylabel("statistics")
    sebplt.ax_add_histogram(
        ax=ax_cnt,
        bin_edges=oa_bin["edges"] ** 2,
        bincounts=h1_cnt,
        linestyle="-",
        linecolor="black",
        linealpha=1.0,
        face_color="black",
        face_alpha=None,
        label=None,
        draw_bin_walls=True,
    )
    ax_add_fov_marker(ax_cnt, FOV_HA_DEG**2)
    ax_cnt.set_ylim([0.0, 1.1 * np.max(h1_cnt)])
    ax_cnt.set_xlim([0.0, OFF_STOP_DEG**2])
    ax_square_format(ax=ax_cnt)
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_counts.jpg"))
    sebplt.close(fig)

    # DISTORTION
    # ==========
    disto = snap["roi_zenith_rad"] / snap["source_zenith_rad"]

    h1_disto_p50 = np.zeros(oa_bin["num"])
    h1_disto_s68 = np.zeros(oa_bin["num"])

    for isi in range(oa_bin["num"]):
        off2_start = oa_bin["edges"][isi]
        off2_stop = oa_bin["edges"][isi + 1]
        off2_mask = np.logical_and(
            psf_off_deg >= off2_start, psf_off_deg < off2_stop
        )
        cnt = np.sum(off2_mask)
        if cnt > 0:
            h1_disto_p50[isi] = np.percentile(disto[off2_mask], 50)
            h1_disto_s68[isi] = percentile_spread(disto[off2_mask], 68)
        else:
            h1_disto_p50[isi] = float("nan")
            h1_disto_s68[isi] = float("nan")

    try:
        distortion_fit, distortion_fit_cov = np.polyfit(
            x=np.rad2deg(snap["source_zenith_rad"]),
            y=np.rad2deg(snap["roi_zenith_rad"]),
            deg=1,
            cov=True,
        )
        distortion_fit_std = np.sqrt(np.diag(distortion_fit_cov))
        distortion_fit_fn_deg = np.poly1d(distortion_fit)
        with rnw.open(
            os.path.join(out_dir, f"{telescope_key:s}_distortion.txt"), "wt"
        ) as f:
            f.write("reco angle / deg = ")
            f.write("(")
            f.write(sci_uncertainty(distortion_fit[0], distortion_fit_std[0]))
            f.write(")")
            f.write(" true angle + ")
            f.write("(")
            f.write(sci_uncertainty(distortion_fit[1], distortion_fit_std[1]))
            f.write(")")
        HAVE_FIT = True
    except ValueError:
        HAVE_FIT = False

    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.05, 0.75, 0.9])
    ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=oa_bin["edges"] ** 2,
        y=h1_disto_p50,
        y_std=h1_disto_s68,
        weights=relative_cnt,
        color="black",
    )
    if SHOW_FIT and HAVE_FIT:
        ax.axhline(y=distortion_fit[0], color="r")

    ax_add_fov_marker(ax, FOV_HA_DEG**2)
    ax.set_xlim([0.0, FOV_HA_DEG**2])
    ax.set_ylabel(r"distortion / 1")
    ax_blank_format(ax=ax)
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_distortion.jpg"))
    sebplt.close(fig)

    # ENERGY
    # ======
    enecon = snap["feed_horn_energy_conservation_ratio"]

    h1_enecon_p50 = np.zeros(oa_bin["num"])
    h1_enecon_s68 = np.zeros(oa_bin["num"])
    h1_cnt = np.zeros(oa_bin["num"])

    for isi in range(oa_bin["num"]):
        off2_start = oa_bin["edges"][isi]
        off2_stop = oa_bin["edges"][isi + 1]
        off2_mask = np.logical_and(
            psf_off_deg >= off2_start, psf_off_deg < off2_stop
        )
        cnt = np.sum(off2_mask)
        h1_cnt[isi] = cnt
        if cnt > 0:
            h1_enecon_p50[isi] = np.percentile(enecon[off2_mask], 50)
            h1_enecon_s68[isi] = percentile_spread(enecon[off2_mask], 68)
        else:
            h1_enecon_p50[isi] = float("nan")
            h1_enecon_s68[isi] = float("nan")

    enecon_lim = [0.0, 2]
    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.05, 0.75, 0.9])
    ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=oa_bin["edges"] ** 2,
        y=h1_enecon_p50,
        y_std=h1_enecon_s68,
        weights=relative_cnt,
        color="black",
    )
    ax_add_fov_marker(ax, FOV_HA_DEG**2)
    ax.set_ylim(enecon_lim)
    ax.set_xlim([0.0, OFF_STOP_DEG**2])
    ax.set_xlabel(XLABEL_OFF_AXIS_DEG2)
    ax.set_ylabel("energy transport\n camera / mirror")
    ax_blank_format(ax=ax)
    fig.savefig(
        os.path.join(out_dir, f"{telescope_key:s}_energy_conservation.jpg")
    )
    sebplt.close(fig)
