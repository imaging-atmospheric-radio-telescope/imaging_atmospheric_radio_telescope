import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import spherical_coordinates
import binning_utils

scenario_key = "fully_outside_field_of_view"

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

            brightest_feed_horn_index = np.argmax(power_ratio)
            response_index = int(os.path.basename(response_path))

            m2deg = lambda x: -np.rad2deg(
                np.arctan(x / telescope["mirror"]["focal_length_m"])
            )

            fig = sebplt.figure(
                style={"rows": 1920, "cols": 1920, "fontsize": 2.0}
            )
            AXES_MINIMAL = {
                "spines": ["left", "bottom"],
                "axes": ["x", "y"],
                "grid": False,
            }
            AXES_BLANK = {"spines": [], "axes": [], "grid": False}
            ax = sebplt.add_axes(
                fig=fig, span=[0.0, 0.0, 1, 1], style=AXES_BLANK
            )
            norm = sebplt.matplotlib.colors.LogNorm(
                vmin=1e-3,
                vmax=1e-1,
            )
            im = iaat.camera.ax_add_camera_feed_horn_scatter_values(
                ax=ax,
                camera=telescope["sensor"],
                feed_horn_scatter_values=response.energy_feed_horns_scatter
                / energy_expected_from_source_J,
                cmap="Blues",
                norm=norm,
                scale_function=m2deg,
            )
            iaat.camera.ax_add_camera_feed_horn_edges(
                ax=ax,
                camera=telescope["sensor"],
                color="black",
                alpha=0.33,
                linewidth=0.2,
                scale_function=m2deg,
            )
            ax.set_xlabel(r"x / (1$^{\circ}$)")
            ax.set_ylabel(r"y / (1$^{\circ}$)")
            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{telescope_key:s}_{response_index:06d}_ghosting.jpg",
                )
            )
            sebplt.close(fig)

            for orientation in ["vertical", "horizontal"]:
                iaat_plot.write_figure_colorbar(
                    path=os.path.join(
                        out_dir,
                        f"{telescope_key:s}_{response_index:06d}_colorbar_{orientation:s}.jpg",
                    ),
                    label=r"energy ratio / 1",
                    norm=norm,
                    orientation=orientation,
                )

            max_source_zenith_angle_rad = (
                4.0 * fov["field_of_view_fully_outside_half_angle_rad"]
            )
            max_source_zenith_angle_deg = np.rad2deg(
                max_source_zenith_angle_rad
            )
            fig = sebplt.figure(
                style={"rows": 1920, "cols": 1920, "fontsize": 2.0}
            )
            ax = sebplt.add_axes(
                fig=fig, span=[0.0, 0.0, 1, 1], style=AXES_BLANK
            )
            ax.set_xlim(
                [-max_source_zenith_angle_deg, max_source_zenith_angle_deg]
            )
            ax.set_ylim(
                [-max_source_zenith_angle_deg, max_source_zenith_angle_deg]
            )
            ax.set_xlabel(r"x / (1$^{\circ}$)")
            ax.set_ylabel(r"y / (1$^{\circ}$)")
            source_cx, source_cy = spherical_coordinates.az_zd_to_cx_cy(
                azimuth_rad=plane_wave_config["geometry"]["azimuth_rad"],
                zenith_rad=plane_wave_config["geometry"]["zenith_rad"],
            )
            ax.plot(
                np.rad2deg(source_cx),
                np.rad2deg(source_cy),
                marker="*",
                markersize=10.0,
                color="black",
            )
            im = iaat.camera.ax_add_camera_feed_horn_values(
                ax=ax,
                camera=telescope["sensor"],
                feed_horn_values=response.energy_feed_horns
                / energy_expected_from_source_J,
                scale_function=m2deg,
                cmap="Blues",
                norm=norm,
            )
            nnn_deg = int(
                np.ceil(np.rad2deg(fov["field_of_view_half_angle_rad"]))
            )
            NNN_deg = int(np.ceil(max_source_zenith_angle_deg))
            for ideg in np.arange(nnn_deg, NNN_deg):
                sebplt.ax_add_circle(
                    ax=ax,
                    x=0.0,
                    y=0.0,
                    r=ideg,
                    color="black",
                    linewidth=0.25,
                    alpha=0.25,
                )
            for iaz_rad in np.arange(0, 2 * np.pi, np.pi / 6):
                _ax = np.cos(iaz_rad)
                _ay = np.sin(iaz_rad)
                ax.plot(
                    _ax * np.array([nnn_deg, NNN_deg - 1]),
                    _ay * np.array([nnn_deg, NNN_deg - 1]),
                    color="black",
                    linewidth=0.25,
                    alpha=0.25,
                )
            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{telescope_key:s}_{response_index:06d}_geometry.jpg",
                )
            )
            sebplt.close(fig)

    azimuths_rad = np.array(azimuths_rad)
    zeniths_rad = np.array(zeniths_rad)
    polarizations_rad = np.array(polarizations_rad)
    power_ratios = np.array(power_ratios)

    _num_bins = int(2 * np.sqrt(np.prod(power_ratios.shape)))
    power_ratio_bin = binning_utils.Binning(
        bin_edges=np.geomspace(1e-5, 1.0, _num_bins)
    )

    bin_counts = np.histogram(power_ratios, bins=power_ratio_bin["edges"])[0]
    relbin_counts = bin_counts / np.sum(bin_counts)

    fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.8, 0.8])
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
    ax.set_xlabel("energy ratio / 1")
    ax.set_ylabel("relative intensity / 1")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_power_ratio.jpg"))
    sebplt.close(fig)
