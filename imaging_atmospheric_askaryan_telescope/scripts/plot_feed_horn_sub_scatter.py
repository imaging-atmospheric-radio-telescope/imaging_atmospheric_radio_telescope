import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np


parser = argparse.ArgumentParser(
    prog="plot_tfeed_horn_sub_scatter.py",
    description=("Plot feed_horn_sub_scatter."),
)
parser.add_argument(
    "--out_dir",
    metavar="OUT_DIR",
    default="feed_horn_sub_scatter",
    type=str,
    help="Path to write figures to.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


args = parser.parse_args()
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

inner_radius_m = 0.2
for num in [1, 3, 4, 5, 6, 7]:

    print(num)
    fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
    sebplt.ax_add_hexagon(
        ax=ax,
        x=0.0,
        y=0.0,
        r_outer=np.sqrt(3) / 2 * inner_radius_m,
        orientation_deg=30,
        color="black",
        linewidth=0.2,
    )
    xy = iaat.telescope.make_feed_horn_sub_scatter(
        num=num, inner_radius_m=inner_radius_m
    )
    print(xy)
    ax.plot(xy[:, 0], xy[:, 1], marker="o", color="red", linewidth=0.0)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, f"{num:d}.jpg"))
    sebplt.close(fig)
