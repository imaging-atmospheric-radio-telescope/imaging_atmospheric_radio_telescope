import argparse
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import json_utils


parser = argparse.ArgumentParser(
    prog="plot_compare_ray_tracing_optics.py",
    description=("Plot comparison with ray tracing."),
)
parser.add_argument(
    "psf_dir",
    metavar="PSF_DIR",
    type=str,
    help="Path to point spread function investigation directory",
)
parser.add_argument(
    "ray_dir",
    metavar="RAY_DIR",
    type=str,
    help="Ray tracing directory",
)
parser.add_argument(
    "--out_dir",
    metavar="OUT_DIR",
    default="plot_compare_ray_tracing_optics",
    type=str,
    help="Path to write figures to.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


args = parser.parse_args()
out_dir = args.out_dir
psf_dir = args.psf_dir
ray_dir = args.ray_dir
os.makedirs(out_dir, exist_ok=True)


def read_json(path):
    with open(path, "rt") as f:
        out = json_utils.loads(f.read())
    return out


TELESCOPE_KEYS = ["crome", "medium_size_telescope", "large_size_telescope"]

for telescope_key in TELESCOPE_KEYS:
    tele_dir = os.path.join(ray_dir, telescope_key)

    telescope = iaat.run.from_config(work_dir=tele_dir)["telescope"]
    fov = iaat.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    airy_radius_m = iaat.telescope.calculate_airy_disk_radius_in_focal_plane(
        telescope=telescope
    )
    airy_area_m2 = np.pi * airy_radius_m**2
    AREA_SCALE = 1 / airy_area_m2

    FOV_HA_DEG = np.rad2deg(fov["field_of_view_half_angle_rad"])
    OFF_STOP_DEG = np.sqrt(np.ceil(FOV_HA_DEG**2))

    # LOAD SUMMARIES
    # ==============
    wav = read_json(
        os.path.join(
            psf_dir,
            "plots",
            "point_spread_function",
            f"{telescope_key:s}.summary.json",
        )
    )
    wav_energy_scale_factor = read_json(
        os.path.join(
            psf_dir,
            "calibration",
            telescope_key,
            "energy_conservation_scale_factor.json",
        )
    )["fitted_energy_scale_factor"]

    ray = read_json(os.path.join(ray_dir, f"{telescope_key:s}.summary.json"))

    relative_cnt = np.sqrt(wav["energy_conservation_1"]["hist"]["cnt"])
    relative_cnt = relative_cnt / np.max(relative_cnt)
    relative_cnt[relative_cnt > 1] = 1.0

    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.05, 0.75, 0.9])
    iaat.investigations.point_spread_function.plot.ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=wav["off_axis_bin_deg"]["edges"] ** 2,
        y=wav["point_spread_function_m2"]["hist"]["p50"] * AREA_SCALE,
        y_std=wav["point_spread_function_m2"]["hist"]["s68"] * AREA_SCALE,
        weights=relative_cnt,
        color="black",
    )
    iaat.investigations.point_spread_function.plot.ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=ray["off_axis_bin_deg"]["edges"] ** 2,
        y=ray["point_spread_function_m2"]["hist"]["p50"] * AREA_SCALE,
        y_std=ray["point_spread_function_m2"]["hist"]["s68"] * AREA_SCALE,
        weights=relative_cnt,
        color="blue",
    )
    iaat.investigations.point_spread_function.plot.ax_add_fov_marker(
        ax, FOV_HA_DEG**2
    )
    # ylim = [0.0, 1.25 * np.nanmax(h_psf_area["p50"]) * AREA_SCALE]
    # ax.set_ylim(ylim)
    ax.set_xlim([0.0, OFF_STOP_DEG**2])
    iaat.investigations.point_spread_function.plot.ax_blank_format(ax=ax)
    ax.set_ylabel("area containing 80% /\n" + "Airy disk")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_spread.jpg"))
    sebplt.close(fig)

    enecon_lim = [0.0, 1.25]
    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.05, 0.75, 0.9])
    iaat.investigations.point_spread_function.plot.ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=wav["off_axis_bin_deg"]["edges"] ** 2,
        y=wav["energy_conservation_1"]["hist"]["p50"]
        * wav_energy_scale_factor,
        y_std=wav["energy_conservation_1"]["hist"]["s68"]
        * wav_energy_scale_factor,
        weights=relative_cnt,
        color="black",
    )
    iaat.investigations.point_spread_function.plot.ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=ray["off_axis_bin_deg"]["edges"] ** 2,
        y=ray["energy_conservation_1"]["hist"]["p50"],
        y_std=ray["energy_conservation_1"]["hist"]["s68"],
        weights=relative_cnt,
        color="blue",
    )
    iaat.investigations.point_spread_function.plot.ax_add_fov_marker(
        ax, FOV_HA_DEG**2
    )
    ax.set_ylim(enecon_lim)
    ax.set_xlim([0.0, OFF_STOP_DEG**2])
    ax.set_ylabel("energy conservation / 1")
    iaat.investigations.point_spread_function.plot.ax_blank_format(ax=ax)
    fig.savefig(
        os.path.join(out_dir, f"{telescope_key:s}_energy_conservation.jpg")
    )
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.05, 0.75, 0.9])
    iaat.investigations.point_spread_function.plot.ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=wav["off_axis_bin_deg"]["edges"] ** 2,
        y=wav["distortion_1"]["hist"]["p50"],
        y_std=wav["distortion_1"]["hist"]["s68"],
        weights=relative_cnt,
        color="black",
    )
    iaat.investigations.point_spread_function.plot.ax_add_uncertain_bins(
        ax=ax,
        x_bin_edges=ray["off_axis_bin_deg"]["edges"] ** 2,
        y=ray["distortion_1"]["hist"]["p50"],
        y_std=ray["distortion_1"]["hist"]["s68"],
        weights=relative_cnt,
        color="blue",
    )
    iaat.investigations.point_spread_function.plot.ax_add_fov_marker(
        ax, FOV_HA_DEG**2
    )
    ax.set_xlim([0.0, FOV_HA_DEG**2])
    ax.set_ylabel(r"distortion / 1")
    iaat.investigations.point_spread_function.plot.ax_blank_format(ax=ax)
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_distortion.jpg"))
    sebplt.close(fig)
