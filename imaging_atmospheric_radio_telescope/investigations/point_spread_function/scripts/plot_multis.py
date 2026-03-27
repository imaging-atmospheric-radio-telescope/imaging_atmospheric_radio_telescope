import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_radio_telescope as iart
from imaging_atmospheric_radio_telescope import plot as iaat_plot
import numpy as np
import spherical_coordinates
import binning_utils
import glob

scenario_key = "multis"

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

config = iart.investigations.point_spread_function.utils.read_config(psf_dir)

prng = np.random.Generator(np.random.PCG64(5))

GIGA = 1e9

source_keyes = ["0", "1"]
source_plot = {
    "0": {"linestyle": "-", "color": "gray"},
    "1": {"linestyle": "--", "color": "blue"},
}
for telescope_key in config["stars"]["telescopes"]:

    telescope, site, timing = (
        iart.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            work_dir=psf_dir, config=config, telescope_key=telescope_key
        )
    )
    fov = iart.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    all_response_paths = iart.utils.filter_integer_filenames(
        paths=glob.glob(os.path.join(psf_dir, "multis", telescope_key, "*"))
    )
    all_response_paths = sorted(all_response_paths)
    num_choice = np.min([12, len(all_response_paths)])
    response_paths_choices = prng.choice(
        len(all_response_paths), size=num_choice, replace=False
    )
    response_paths = []
    for response_paths_choice in response_paths_choices:
        response_paths.append(all_response_paths[response_paths_choice])

    r_roi = 4 * telescope["sensor"]["camera"]["feed_horn_inner_radius_m"]

    nu_start_Hz, nu_stop_Hz = iart.lownoiseblock.input_frequency_start_stop_Hz(
        telescope["lnb"]
    )
    frequency_bin = binning_utils.Binning(
        bin_edges=np.linspace(nu_start_Hz - 250e6, nu_stop_Hz + 250e6, 37)
    )

    for response_path in response_paths:
        response_id = int(os.path.basename(response_path))
        response = iart.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )

        masks = {}
        expected_energies_J = {}
        expected_frequency_Hz = {}
        expected_screen = {}
        for skey in source_keyes:
            s_config = response.source_config["plane_waves"][skey]

            expected_energies_J[skey] = (
                iart.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
                    config=s_config,
                    area_m2=telescope["mirror"]["area_m2"],
                )
            )
            expected_frequency_Hz[skey] = s_config["sine_wave"][
                "emission_frequency_Hz"
            ]
            expected_screen[skey] = (
                iart.utils.sky_and_screen.sky_az_zd_to_screen_x_y(
                    azimuth_rad=s_config["geometry"]["azimuth_rad"],
                    zenith_rad=s_config["geometry"]["zenith_rad"],
                    focal_length_m=telescope["mirror"]["focal_length_m"],
                )
            )
            masks[skey] = (
                iart.investigations.point_spread_function.plane_wave_response.mask_feed_horns(
                    feed_horn_positions_m=telescope["sensor"][
                        "feed_horn_positions_m"
                    ],
                    containment_radius_m=r_roi,
                    azimuth_rad=s_config["geometry"]["azimuth_rad"],
                    zenith_rad=s_config["geometry"]["zenith_rad"],
                )
            )

        masks_have_no_overlap = np.logical_not(
            np.any(np.logical_and(masks["0"], masks["1"]))
        )

        if masks_have_no_overlap:
            received_energies_J = {}
            power_densities_W_per_Hz = {}
            for skey in source_keyes:
                received_energies_J[skey] = np.sum(
                    response.energy_feed_horns[masks[skey]]
                )

                E_feed_horns_roi = iart.time_series.zeros_like(
                    response.E_feed_horns, num_channels=np.sum(masks[skey])
                )
                E_feed_horns_roi[:] = response.E_feed_horns[masks[skey]]

                power_density_W_per_Hz_per_m2 = iart.electric_fields.estimate_power_spectrum_density_W_per_Hz_per_m2(
                    electric_fields=E_feed_horns_roi,
                    antenna_effective_area_m2=telescope["sensor"][
                        "feed_horn_area_m2"
                    ],
                    frequency_bin_edges_Hz=frequency_bin["edges"],
                    components=[True, True, True],
                )
                power_densities_W_per_Hz[skey] = telescope["sensor"][
                    "feed_horn_area_m2"
                ] * np.mean(power_density_W_per_Hz_per_m2, axis=1)

            dmax = 1.1 * np.max(
                [power_densities_W_per_Hz["0"], power_densities_W_per_Hz["1"]]
            )
            rdmin = 1e-3
            fig = sebplt.figure(
                style={"rows": 1080, "cols": 1920, "fontsize": 2.0}
            )
            ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
            for skey in source_keyes:
                ax.axvline(
                    expected_frequency_Hz[skey] / GIGA,
                    color="gray",  # source_plot[skey]["color"],
                    linestyle=source_plot[skey]["linestyle"],
                    linewidth=2.0,
                )
                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=frequency_bin["edges"] / GIGA,
                    bincounts=power_densities_W_per_Hz[skey],
                    bincounts_upper=power_densities_W_per_Hz[skey],
                    bincounts_lower=rdmin * power_densities_W_per_Hz[skey],
                    linestyle=source_plot[skey]["linestyle"],
                    linecolor="k",
                    linealpha=1.0,
                    face_alpha=0.5,
                    face_color="gray",  # source_plot[skey]["color"],
                    draw_bin_walls=True,
                )
            ax.set_ylim([rdmin * dmax, dmax])
            ax.semilogy()
            ax.set_xlim(frequency_bin["limits"] / GIGA)
            ax.set_xlabel(r"frequency $\nu$ / GHz")
            ax.set_ylabel(r"power density / W (Hz)$^{-1}$")
            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{telescope_key:s}_{response_id:06d}_power_spectrum.jpg",
                )
            )
            sebplt.close(fig)

            feed_horn_energies_eV = (
                response.energy_feed_horns / iart.signal.ELECTRON_VOLT_J
            )
            vmax = np.max(feed_horn_energies_eV)
            norm = sebplt.matplotlib.colors.PowerNorm(
                vmin=1e-2 * vmax,
                vmax=vmax,
                gamma=0.5,
            )

            for orientation in ["vertical", "horizontal"]:
                iaat_plot.write_figure_colorbar(
                    path=os.path.join(
                        out_dir,
                        f"{telescope_key:s}_{response_id:06d}_colorbar_{orientation:s}.jpg",
                    ),
                    label=r"energy / eV",
                    norm=norm,
                    orientation=orientation,
                )

            fig = sebplt.figure(
                style={"rows": 1920, "cols": 1920, "fontsize": 2.0}
            )
            ax = sebplt.add_axes(
                fig=fig, span=[-0.05, -0.05, 1.1, 1.1], style=sebplt.AXES_BLANK
            )
            iart.camera.ax_add_camera_feed_horn_values(
                ax=ax,
                camera=telescope["sensor"],
                feed_horn_values=feed_horn_energies_eV,
                scale_function=None,
                cmap="Blues",
                norm=norm,
            )
            iart.camera.ax_add_camera_feed_horn_edges(
                ax=ax,
                camera=telescope["sensor"],
                color="black",
                alpha=0.1,
                linewidth=0.5,
            )
            for skey in source_keyes:
                sebplt.ax_add_circle(
                    ax=ax,
                    x=expected_screen[skey][0],
                    y=expected_screen[skey][1],
                    r=r_roi,
                    color="black",
                    linestyle=source_plot[skey]["linestyle"],
                    linewidth=2,
                    alpha=1,
                )
            fig.savefig(
                os.path.join(
                    out_dir, f"{telescope_key:s}_{response_id:06d}_image.jpg"
                )
            )
            sebplt.close(fig)

            fig = sebplt.figure(
                style={"rows": 120, "cols": 1920, "fontsize": 2.0}
            )
            ax = sebplt.add_axes(
                fig=fig, span=[0, 0, 1.0, 1.0], style=sebplt.AXES_BLANK
            )
            _x = 0
            for skey in source_keyes:
                e_eV = expected_energies_J[skey] / iart.signal.ELECTRON_VOLT_J
                n_GHZ = expected_frequency_Hz[skey] / GIGA
                ax.plot(
                    _x + np.array([0.025, 0.125]),
                    [0.5, 0.5],
                    linestyle=source_plot[skey]["linestyle"],
                    color="black",
                    linewidth=2,
                    transform=ax.transAxes,
                )
                ax.text(
                    x=_x + 0.15,
                    y=0.5,
                    s=f"{e_eV:.2f}eV, {n_GHZ:.2f}GHz",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="center",
                    horizontalalignment="left",
                )
                _x += 0.5
            fig.savefig(
                os.path.join(
                    out_dir, f"{telescope_key:s}_{response_id:06d}_legend.jpg"
                )
            )
            sebplt.close(fig)
