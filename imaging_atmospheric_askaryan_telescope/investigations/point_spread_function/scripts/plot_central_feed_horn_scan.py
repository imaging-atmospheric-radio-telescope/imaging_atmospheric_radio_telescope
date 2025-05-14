import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import glob
import binning_utils

scenario_key = "central_feed_horn_scan"

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
for telescope_key in config["stars"]["telescopes"]:
    telescope, site, timing = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            config=config, telescope_key=telescope_key
        )
    )
    fov = iaat.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    fh_index = iaat.camera.get_index_of_central_feed_horn(
        camera=telescope["sensor"]
    )

    response_paths = (
        iaat.investigations.point_spread_function.stars.list_response_paths(
            work_dir=psf_dir,
            telescope_key=telescope_key,
            scenario_key=scenario_key,
        )
    )

    off_axis_angles_rad = []
    power_ratios = []

    for response_path in response_paths:
        response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )

        plane_wave_config = response.source_config["plane_waves"][source_key]

        energy_in_central_feed_horn_J = response.energy_feed_horns[fh_index]
        energy_expected_from_source_J = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
            config=plane_wave_config,
            area_m2=telescope["mirror"]["area_m2"],
        )
        off_axis_angle_rad = plane_wave_config["geometry"]["zenith_rad"]
        power_ratio = (
            energy_in_central_feed_horn_J / energy_expected_from_source_J
        )

        off_axis_angles_rad.append(off_axis_angle_rad)
        power_ratios.append(power_ratio)

    off_axis_angles_rad = np.array(off_axis_angles_rad)
    power_ratios = np.array(power_ratios)

    asort = np.argsort(off_axis_angles_rad)
    off_axis_angles_rad = off_axis_angles_rad[asort]
    power_ratios = power_ratios[asort]

    off_axis_bin = binning_utils.Binning(
        bin_edges=np.linspace(
            0.0,
            4.0 * fov["central_feed_horn_half_angle_rad"],
            int(np.ceil(2 * np.sqrt(len(off_axis_angles_rad)))),
        )
    )
    bincounts = np.histogram(off_axis_angles_rad, bins=off_axis_bin["edges"])[
        0
    ]
    binweigths = np.histogram(
        off_axis_angles_rad, bins=off_axis_bin["edges"], weights=power_ratios
    )[0]
    bincounts_relunc = np.sqrt(bincounts) / bincounts
    binweigths_absunc = binweigths * bincounts_relunc

    binned_power_ratio = binweigths / bincounts
    binned_power_ratio_absunc = binweigths_absunc / bincounts

    ylim = [1e-4, 2]
    fig = sebplt.figure(style={"rows": 720, "cols": 1920, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.25, 0.65, 0.65])
    ax.plot(
        np.rad2deg(off_axis_angles_rad),
        power_ratios,
        color="black",
        marker=".",
        alpha=0.33,
    )
    ax.vlines(
        x=np.rad2deg(fov["central_feed_horn_half_angle_rad"]),
        ymin=ylim[0],
        ymax=ylim[1],
        linestyle="--",
        color="black",
    )
    ax.semilogy()
    ax.set_ylim(ylim)
    ax.set_xlim(np.rad2deg(off_axis_bin["limits"]))
    ax.set_xlabel(r"off axis angle / (1$^{\circ}$)")
    ax.set_ylabel("power ratio / 1")
    fig.savefig(
        os.path.join(out_dir, f"{telescope_key:s}_power_ratio_simple.jpg")
    )
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 720, "cols": 1920, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.25, 0.65, 0.65])
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=np.rad2deg(off_axis_bin["edges"]),
        bincounts=binned_power_ratio,
        bincounts_lower=binned_power_ratio - binned_power_ratio_absunc,
        bincounts_upper=binned_power_ratio + binned_power_ratio_absunc,
        linecolor="black",
        face_color="black",
        face_alpha=0.25,
    )
    ax.vlines(
        x=np.rad2deg(fov["central_feed_horn_half_angle_rad"]),
        ymin=ylim[0],
        ymax=ylim[1],
        linestyle="--",
        color="black",
    )
    ax.semilogy()
    ax.set_ylim(ylim)
    ax.set_xlim(np.rad2deg(off_axis_bin["limits"]))
    ax.set_xlabel(r"off axis angle / (1$^{\circ}$)")
    ax.set_ylabel("power ratio / 1")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_power_ratio.jpg"))
    sebplt.close(fig)
