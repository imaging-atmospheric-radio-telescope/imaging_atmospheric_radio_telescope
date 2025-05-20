import argparse
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import optic_object_wavefronts as oow
import numpy as np
import spherical_histogram
import spherical_coordinates
import solid_angle_utils as sau
import triangle_mesh_io
import thin_lens


parser = argparse.ArgumentParser(
    prog="plot_telescope_mesh.py",
    description=("Plot a 3D mesh of the askaryan telescope."),
)
parser.add_argument(
    "--out_dir",
    metavar="OUT_DIR",
    default="telescope_mesh",
    type=str,
    help="Path to write figures to.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


args = parser.parse_args()
out_dir = args.out_dir


def ax_add_important_rings(ax, telescope):
    sebplt.ax_add_circle(
        ax=ax,
        x=0,
        y=0,
        r=telescope["mirror"]["inner_radius_m"],
        color="black",
        linestyle="-",
        alpha=0.33,
    )
    sebplt.ax_add_circle(
        ax=ax,
        x=0,
        y=0,
        r=telescope["mirror"]["outer_radius_m"],
        color="black",
        linestyle="-",
        alpha=0.33,
    )
    sebplt.ax_add_circle(
        ax=ax,
        x=0,
        y=0,
        r=telescope["sensor"]["camera"]["outer_radius_m"],
        color="black",
        linestyle="-",
        alpha=0.33,
    )


def ax_add_mirror_xy(ax, telescope, roi_width):
    ax.scatter(
        telescope["mirror"]["scatter_center_positions_m"][:, 0],
        telescope["mirror"]["scatter_center_positions_m"][:, 1],
        color="black",
        linewidth=0.0,
        alpha=0.66,
        edgecolor="none",
    )


def ax_add_camera_xy(ax, telescope, roi_width):
    camera = telescope["sensor"]

    fh_r = 10 * np.sqrt(camera["feed_horn_area_m2"]) / roi_width
    print(fh_r)

    iaat.camera.ax_add_camera_feed_horn_edges(
        ax=ax, camera=camera, color="lightgray", linewidth=1 * fh_r
    )

    iaat.camera.ax_add_camera_feed_horn_scatter_centers(
        ax=ax,
        camera=camera,
        marker="o",
        alpha=0.66,
        markersize=5 * fh_r,
        markeredgewidth=0.0,
        color="black",
    )


FIGSIZE = {"rows": 1920, "cols": 1920, "fontsize": 2.0}
TELESCOPE_KEYS = [
    "crome",
    "medium_size_telescope",
    "large_size_telescope",
]

for telescope_key in TELESCOPE_KEYS:
    tele_dir = os.path.join(out_dir, telescope_key)

    iaat.run.init(work_dir=tele_dir, telescope_key=telescope_key)
    telescope = iaat.run.from_config(work_dir=tele_dir)["telescope"]

    close_up_items = {
        "camera": {
            "range": 1.1 * telescope["sensor"]["camera"]["outer_radius_m"]
        },
        "mirror": {"range": 1.1 * telescope["mirror"]["outer_radius_m"]},
    }

    for item in close_up_items:
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.8, 0.8])

        ax_add_important_rings(ax=ax, telescope=telescope)

        if item == "mirror":
            ax_add_mirror_xy(
                ax=ax,
                telescope=telescope,
                roi_width=close_up_items["camera"]["range"],
            )
        elif item == "camera":
            ax_add_camera_xy(
                ax=ax,
                telescope=telescope,
                roi_width=close_up_items["camera"]["range"],
            )

        ax.set_xlim([0, close_up_items[item]["range"]])
        ax.set_ylim([0, close_up_items[item]["range"]])

        if item == "mirror":
            sebplt.ax_add_box(
                ax=ax,
                xlim=[0, close_up_items["camera"]["range"]],
                ylim=[0, close_up_items["camera"]["range"]],
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
            )

        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ / m")
        ax.set_ylabel(r"$y$ / m")
        fig.savefig(
            os.path.join(tele_dir, f"telescope_{item:s}_close_up_xy.jpg")
        )
        sebplt.close(fig)

    f_over_D = telescope["mirror"]["focal_length_m"] / (
        2 * telescope["mirror"]["outer_radius_m"]
    )

    fig = sebplt.figure(
        {"rows": 1920 * f_over_D + 200, "cols": 1920, "fontsize": 2.0}
    )
    ax = sebplt.add_axes(
        fig=fig,
        span=[0.15, 0.15, 0.8, 0.8],
        style={"spines": [], "axes": ["x", "y"], "grid": True},
    )

    object_distance = 8e3

    mD = telescope["mirror"]["outer_radius_m"]
    sR = telescope["sensor"]["camera"]["outer_radius_m"]
    miD = telescope["mirror"]["inner_radius_m"]
    mf = telescope["mirror"]["focal_length_m"]
    sensor_d = thin_lens.compute_image_distance_for_object_distance(
        object_distance, mf
    )
    ax.scatter(
        telescope["mirror"]["scatter_center_positions_m"][:, 0],
        telescope["mirror"]["scatter_center_positions_m"][:, 2],
        s=10,
        color="black",
        linewidth=0.0,
        edgecolor="none",
        alpha=0.05,
    )
    ax.scatter(
        telescope["sensor"]["feed_horn_positions_m"][:, 0],
        sensor_d * np.ones(telescope["sensor"]["num_feed_horns"]),
        s=10,
        color="black",
        linewidth=0.0,
        edgecolor="none",
        alpha=0.05,
    )
    ax.plot(
        [-1.1 * mD, +1.1 * mD],
        [0, 0],
        color="black",
        linewidth=0.75,
    )
    ax.plot(
        [0, 0],
        [-0.1 * mf, 1.1 * mf],
        color="black",
        linewidth=0.75,
    )

    ax.plot(
        mD * np.array([-0.5, +0.5, +0.55, 0.6]),
        mf * np.array([1, 1, 0.93, 0.93]),
        color="black",
        linewidth=0.5,
    )
    ax.text(
        s=r"$f$",
        x=0.6 * mD,
        y=mf * 0.93,
        verticalalignment="center",
    )

    ax.plot(
        mD * np.array([-0.5, +0.5, +0.55, 0.6]),
        sensor_d * np.array([1, 1, 1.07, 1.07]),
        color="black",
        linewidth=0.5,
    )
    ax.text(
        s=r"$d$",
        x=+0.6 * mD,
        y=sensor_d * 1.07,
        verticalalignment="center",
    )

    ax.plot(
        [mD, mD],
        [0, mf * 0.25],
        color="black",
        linestyle="--",
        linewidth=0.75,
        alpha=0.5,
    )
    ax.text(
        s=r"$D/2$",
        x=mD * 0.92,
        y=mf * 0.25,
        horizontalalignment="right",
        verticalalignment="center",
    )

    ax.plot(
        [miD, miD],
        [0, mf],
        color="black",
        linestyle="--",
        linewidth=0.75,
        alpha=0.5,
    )
    ax.text(
        s=r"$D_\mathrm{inner}/2$",
        x=miD,
        y=0.5 * mf,
        verticalalignment="center",
    )

    ax.plot(
        [sR, sR],
        [0.95 * sensor_d, 1.1 * mf],
        color="black",
        linestyle="--",
        linewidth=0.75,
        alpha=0.5,
    )
    ax.text(
        s=r"$D_\mathrm{screen}/2$",
        x=sR,
        y=1.1 * sensor_d,
        horizontalalignment="left",
        verticalalignment="center",
    )

    # field of view
    sebplt.ax_add_pie_slice(
        ax=ax,
        x=0,
        y=0,
        phi_start_rad=np.arctan(sR / mf) + np.pi / 2,
        phi_stop_rad=0.0 + np.pi / 2,
        radius=0.9 * mf,
        color="black",
        linestyle="-.",
        fill="none",
        linewidth=0.75,
        alpha=0.15,
    )
    ax.plot(
        [0.0, -2 * sR],
        [0.0, 2 * mf],
        color="black",
        linestyle="-.",
        linewidth=0.75,
        alpha=0.5,
    )
    ax.text(
        s=r"$\Theta_\text{field-of-view}/2$",
        x=-1.4 * sR,
        y=sensor_d * 1.1,
        horizontalalignment="right",
        verticalalignment="center",
    )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ / m")
    ax.set_ylabel(r"$z$ / m")
    ax.set_xlim(np.array([-1.1, 1.1]) * mD)
    ax.set_ylim(np.array([-0.025, 1.1]) * mf)

    fig.savefig(os.path.join(tele_dir, f"telescope_side_scatch.jpg"))
    sebplt.close(fig)
