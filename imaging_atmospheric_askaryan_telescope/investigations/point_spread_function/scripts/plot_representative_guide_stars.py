import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import glob
import binning_utils

scenario_key = "representative_guide_stars"

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

    response_paths = (
        iaat.investigations.point_spread_function.stars.list_response_paths(
            work_dir=psf_dir,
            telescope_key=telescope_key,
            scenario_key=scenario_key,
        )
    )

    radius_airy_m = iaat.telescope.calculate_airy_disk_radius_in_focal_plane(
        telescope=telescope
    )

    snaps = {}

    for response_path in response_paths:
        response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )
        response_index = int(os.path.basename(response_path))

        plane_wave_config = response.source_config["plane_waves"][source_key]

        energy_expected_from_source_J = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
            config=plane_wave_config,
            area_m2=telescope["mirror"]["area_m2"],
        )

        x_bin_edges, y_bin_edges, energy = response.energy_roi(source_key)

        snaps[response_index] = {
            "x_bin_edges": x_bin_edges,
            "y_bin_edges": y_bin_edges,
            "image": energy / energy_expected_from_source_J,
            "azimuth_rad": plane_wave_config["geometry"]["azimuth_rad"],
            "zenith_rad": plane_wave_config["geometry"]["zenith_rad"],
        }

    vmax = 0
    for response_index in snaps:
        if np.max(snaps[response_index]["image"]) > vmax:
            vmax = np.max(snaps[response_index]["image"])
    for response_index in snaps:
        snaps[response_index]["image"] /= vmax

    norm = sebplt.matplotlib.colors.LogNorm(
        vmin=1e-2,
        vmax=1,
    )

    for orientation in ["vertical", "horizontal"]:
        iaat_plot.write_figure_colorbar(
            path=os.path.join(
                out_dir,
                f"{telescope_key:s}_colorbar_{orientation:s}.jpg",
            ),
            label=r"energy ratio / 1",
            norm=norm,
            orientation=orientation,
            wide=1280 * 1.1,
            narrow=720,
            fontsize=2.5,
        )

    for ri in snaps:
        xlim = [
            np.min(snaps[ri]["x_bin_edges"]),
            np.max(snaps[ri]["x_bin_edges"]),
        ]
        x_range = xlim[1] - xlim[0]
        part_xlim = [xlim[0] + x_range * 0.1, xlim[1] - x_range * 0.1]
        fig = sebplt.figure(
            style={"rows": 1280 * 1.1, "cols": 1280, "fontsize": 2.5}
        )
        ax = sebplt.add_axes(
            fig=fig,
            span=[0.0, 0.1, 1, 0.9],
            style={
                "spines": ["left", "bottom"],
                "axes": ["x", "y"],
                "grid": True,
            },
        )
        ax_xlabel = sebplt.add_axes(
            fig=fig,
            span=[0.1, 0.1, 0.8, 0.0],
            style={
                "spines": ["bottom"],
                "axes": ["x"],
                "grid": False,
            },
        )
        im = ax.pcolormesh(
            snaps[ri]["x_bin_edges"],
            snaps[ri]["y_bin_edges"],
            snaps[ri]["image"].T,
            cmap="Blues",
            norm=norm,
        )
        iaat.camera.ax_add_camera_feed_horn_edges(
            ax=ax,
            camera=telescope["sensor"],
            color="black",
            alpha=0.2,
            linewidth=1,
        )
        sebplt.ax_add_circle(
            ax=ax,
            x=np.mean(snaps[ri]["x_bin_edges"]),
            y=np.mean(snaps[ri]["y_bin_edges"]),
            r=radius_airy_m,
            color="Gray",
            linestyle="--",
        )
        ax.plot(
            np.mean(snaps[ri]["x_bin_edges"]),
            np.mean(snaps[ri]["y_bin_edges"]),
            color="Gray",
            marker="x",
            markersize=10,
        )
        ax.set_xlim(xlim)
        ax_xlabel.set_xlim(part_xlim)
        ax.set_ylim(
            [
                np.min(snaps[ri]["y_bin_edges"]),
                np.max(snaps[ri]["y_bin_edges"]),
            ]
        )
        ax.set_xticklabels(["" for i in ax.get_xticklabels()])
        fig.savefig(
            os.path.join(
                out_dir,
                f"{telescope_key:s}_{ri:06d}.jpg",
            )
        )
        sebplt.close(fig)
