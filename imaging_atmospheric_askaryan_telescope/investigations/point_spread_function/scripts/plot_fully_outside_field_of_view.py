import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import binning_utils

scenario_key = "fully_outside_field_of_view"

parser = argparse.ArgumentParser(
    prog=f"plot_{scenario_key:s}.py",
    description=("Plot feed_horn_sub_scatter."),
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

    response_paths = (
        iaat.investigations.point_spread_function.stars.list_response_paths(
            work_dir=psf_dir,
            telescope_key=telescope_key,
            scenario_key=scenario_key,
        )
    )

    azimuths_rad = []
    zeniths_rad = []
    polarizations_rad = []
    power_ratios = []

    for response_path in response_paths:
        response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )
        plane_wave_config = response.source_config["plane_waves"][source_key]

        energy_expected_from_source_J = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
            config=plane_wave_config,
            area_m2=telescope["mirror"]["area_m2"],
        )
        power_ratio = (
            response.energy_feed_horns / energy_expected_from_source_J
        )

        azimuths_rad.append(plane_wave_config["geometry"]["azimuth_rad"])
        zeniths_rad.append(plane_wave_config["geometry"]["zenith_rad"])
        polarizations_rad.append(
            plane_wave_config["geometry"]["polarization_angle_rad"]
        )
        power_ratios.append(power_ratio)

        if np.any(power_ratio > power_ratio_threshold):
            print(response_path)

    azimuths_rad = np.array(azimuths_rad)
    zeniths_rad = np.array(zeniths_rad)
    polarizations_rad = np.array(polarizations_rad)
    power_ratios = np.array(power_ratios)

    power_ratio_bin = binning_utils.Binning(
        bin_edges=np.geomspace(
            1e-5, 1.0, int(2 * np.sqrt(np.prod(power_ratios.shape)))
        )
    )

    bin_counts = np.histogram(power_ratios, bins=power_ratio_bin["edges"])[0]
    relbin_counts = bin_counts / np.sum(bin_counts)

    fig = sebplt.figure(style={"rows": 720, "cols": 1920, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.25, 0.65, 0.65])
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=power_ratio_bin["edges"],
        bincounts=relbin_counts,
        linecolor="black",
        face_color="black",
        face_alpha=0.25,
    )
    ax.loglog()
    ax.set_ylim([1e-5, 1e-0])
    ax.set_xlim(power_ratio_bin["limits"])
    ax.set_xlabel("power ratio / 1")
    ax.set_ylabel("relative intensity / 1")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_power_ratio.jpg"))
    sebplt.close(fig)
