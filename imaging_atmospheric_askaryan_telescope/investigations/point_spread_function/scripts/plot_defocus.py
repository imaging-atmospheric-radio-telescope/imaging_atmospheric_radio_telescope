import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import glob
import binning_utils
import pandas

scenario_key = "defocus"

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
prng = np.random.Generator(np.random.PCG64(5))


def records_to_recarray(a):
    df = pandas.DataFrame.from_records(a)
    return df.to_records(index=False)


for telescope_key in config["defocus"]["telescopes"]:

    Q = []

    telescope, site, timing = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            work_dir=psf_dir, config=config, telescope_key=telescope_key
        )
    )
    fov = iaat.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )

    response_paths = glob.glob(
        os.path.join(psf_dir, "defocus", telescope_key, "*")
    )

    feed_horn_radial_distance_m = np.linalg.norm(
        telescope["sensor"]["feed_horn_positions_m"][:, 0:2], axis=1
    )
    feed_horn_energies_J = []

    for response_path in response_paths:
        response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
            response_path
        )

        plane_wave_config = response.source_config["plane_waves"][source_key]
        energy_expected_from_source_J = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
            config=plane_wave_config,
            area_m2=telescope["mirror"]["area_m2"],
        )

        rec = {}
        rec["id"] = int(os.path.basename(response.path))
        rec["energy_expected_from_source_J"] = energy_expected_from_source_J
        rec["sensor_distance_m"] = response.sensor["sensor_distance_m"]
        rec["energy_in_feed_horns_J"] = np.sum(response.energy_feed_horns)
        feed_horn_energies_J.append(response.energy_feed_horns)
        Q.append(rec)

    Q = records_to_recarray(Q)
    feed_horn_energies_J = np.array(feed_horn_energies_J)

    args = np.argsort(Q["sensor_distance_m"])
    Q = Q[args]
    feed_horn_energies_J = feed_horn_energies_J[args]

    f_mirror = telescope["mirror"]["focal_length_m"]
    enecon_lim = [0.0, 2]
    fig = sebplt.figure(style={"rows": 960, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        Q["sensor_distance_m"] / f_mirror,
        Q["energy_in_feed_horns_J"] / Q["energy_expected_from_source_J"],
        color="black",
        alpha=0.2,
    )
    ax.plot(
        Q["sensor_distance_m"] / f_mirror,
        Q["energy_in_feed_horns_J"] / Q["energy_expected_from_source_J"],
        color="black",
        alpha=0.2,
        marker="o",
        linewidth=0.0,
    )
    ax.set_ylim(enecon_lim)
    ax.set_xlim(
        np.array(
            [
                config["defocus"]["start_sensor_distance_f"],
                config["defocus"]["stop_sensor_distance_f"],
            ]
        )
    )
    ax.set_xlabel(r"sensor distance $d$ / f")
    ax.set_ylabel("energy transport\n camera / mirror")
    fig.savefig(
        os.path.join(out_dir, f"{telescope_key:s}_energy_conservation.jpg")
    )
    sebplt.close(fig)

    R_outer_m = telescope["mirror"]["outer_radius_m"]
    R_inner_m = telescope["mirror"]["inner_radius_m"]

    feed_horn_energies_eV = feed_horn_energies_J / iaat.signal.ELECTRON_VOLT_J
    vmax = np.max(feed_horn_energies_eV)
    norm = sebplt.matplotlib.colors.LogNorm(
        vmin=1e-3 * vmax,
        vmax=vmax,
    )

    for orientation in ["vertical", "horizontal"]:
        iaat_plot.write_figure_colorbar(
            path=os.path.join(
                out_dir,
                f"{telescope_key:s}_colorbar_{orientation:s}.jpg",
            ),
            label=r"energy / eV",
            norm=norm,
            orientation=orientation,
        )

    choice = prng.choice(a=len(Q), size=8, replace=False)

    for iq in choice:

        sensor_m = Q["sensor_distance_m"][iq]
        scale_factor = np.abs(1 - (sensor_m / f_mirror))
        r_inner = R_inner_m * scale_factor
        r_outer = R_outer_m * scale_factor

        fig = sebplt.figure(
            style={"rows": 1920, "cols": 1920, "fontsize": 2.0}
        )
        ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.89, 0.89])
        iaat.camera.ax_add_camera_feed_horn_values(
            ax=ax,
            camera=telescope["sensor"],
            feed_horn_values=feed_horn_energies_eV[iq],
            scale_function=None,
            cmap="Blues",
            norm=norm,
        )
        iaat.camera.ax_add_camera_feed_horn_edges(
            ax=ax,
            camera=telescope["sensor"],
            color="black",
            alpha=0.1,
            linewidth=0.5,
        )
        sebplt.ax_add_circle(
            ax=ax,
            x=0.0,
            y=0.0,
            r=r_outer,
            color="black",
            linewidth=1,
            alpha=1,
        )
        sebplt.ax_add_circle(
            ax=ax,
            x=0.0,
            y=0.0,
            r=r_inner,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=1,
        )
        ax.text(
            x=0.05,
            y=0.05,
            s=r"$d$ = " + f"{sensor_m/f_mirror:.3f}" + r"$f$",
            transform=ax.transAxes,
            fontsize=16,
        )
        fig.savefig(
            os.path.join(
                out_dir, f"{telescope_key:s}_example_{Q['id'][iq]:06d}.jpg"
            )
        )
        sebplt.close(fig)

    """
    radial_bin = binning_utils.Binning(
        bin_edges=np.linspace(
            0, telescope["sensor"]["camera"]["outer_radius_m"], 17
        )
    )
    Num_rr = 100
    rr = np.linspace(radial_bin["start"], radial_bin["stop"], Num_rr)

    AXES_STYLE = {"spines": ["bottom"], "axes": [], "grid": False}

    R_outer_m = telescope["mirror"]["outer_radius_m"]
    R_inner_m = telescope["mirror"]["inner_radius_m"]
    fig = sebplt.figure(style={"rows": 2880, "cols": 1920, "fontsize": 2.0})
    II = feed_horn_energies_J.shape[0]
    for ii in range(II):
        sensor_m = Q["sensor_distance_m"][ii]
        scale_factor = np.abs(1 - (sensor_m / f_mirror))
        r_inner = R_inner_m * scale_factor
        r_outer = R_outer_m * scale_factor

        expected = np.zeros(Num_rr)
        for jj in range(Num_rr):
            if rr[jj] >= r_inner and rr[jj] <= r_outer:
                expected[jj] = 1
        expected = expected / np.sum(expected)

        print(sensor_m, scale_factor)
        hstep = 0.93 / II
        ax = sebplt.add_axes(
            fig=fig,
            span=[0.2, 0.07 + ii * hstep, 0.75, hstep * 0.9],
            style=AXES_STYLE,
        )
        ii_totoal = np.sum(feed_horn_energies_J[ii])
        his_total_energy = np.histogram(
            feed_horn_radial_distance_m,
            bins=radial_bin["edges"],
            weights=feed_horn_energies_J[ii] / ii_totoal,
        )[0]
        his_num_feed_horns = np.histogram(
            feed_horn_radial_distance_m,
            bins=radial_bin["edges"],
        )[0]
        his = his_total_energy / his_num_feed_horns

        exp_his_total_energy = np.histogram(
            rr, bins=radial_bin["edges"], weights=expected
        )[0]
        exp_his_num_feed_horns = np.histogram(
            rr,
            bins=radial_bin["edges"],
        )[0]
        exp_his = exp_his_total_energy / exp_his_num_feed_horns

        ax.text(
            x=-0.25,
            y=0.5,
            s=f"{sensor_m/f_mirror:.3f}",
            transform=ax.transAxes,
        )

        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=radial_bin["edges"],
            bincounts=his,
            draw_bin_walls=True,
        )

        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=radial_bin["edges"],
            bincounts=exp_his,
            bincounts_upper=exp_his,
            bincounts_lower=1e-4 * np.ones(len(exp_his)),
            linealpha=0.0,
            face_color="Black",
            face_alpha=0.15,
        )

        sebplt.ax_add_grid_with_explicit_ticks(
            ax=ax, xticks=np.linspace(0, 1, 11), yticks=[1e-3, 1e-2, 1e-1, 1]
        )
        ax.semilogy()
        ax.set_xlim(radial_bin["limits"])
        ax.set_ylim([1e-3, 1e0])

    ax_labels = sebplt.add_axes(
        fig=fig,
        span=[0.2, 0.07, 0.75, 0],
        style={"spines": ["bottom"], "axes": ["x"], "grid": False},
    )
    ax_labels.set_xlim(radial_bin["limits"])
    ax_labels.set_xlabel(r"radius / m")
    fig.savefig(os.path.join(out_dir, f"{telescope_key:s}_stack.jpg"))
    sebplt.close(fig)
    """
