import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import optic_object_wavefronts
import posixpath


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

screen_radius_m = 0.1
feed_horn_inner_radius_m = 0.03
focal_ratio_1 = 1.4
feed_horn_oversampling = 1

cc = iaat.camera.make_camera(
    sensor_outer_radius_m=screen_radius_m,
    sensor_distance_m=32.2,
    feed_horn_inner_radius_m=feed_horn_inner_radius_m,
    feed_horn_transmission=0.5,
    feed_horn_focal_ratio_1=focal_ratio_1,
    feed_horn_oversampling=feed_horn_oversampling,
    low_noise_block_effective_area_m2=iaat.signal.calculate_antenna_effective_area(
        wavelength=0.03,
        gain=1.0,
    ),
)

fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
iaat.camera.ax_add_camera(ax=ax, camera=cc, color="black")
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_aspect("equal")
fig.savefig(os.path.join(out_dir, "feed_horn_mesh.jpg"))
sebplt.close(fig)
