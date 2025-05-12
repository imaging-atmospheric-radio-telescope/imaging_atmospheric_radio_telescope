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


screen_radius_m = 0.1
feed_horn_inner_radius_m = 0.03
focal_ratio_1 = 1.4
num_over = 1

# start
feed_horn_outer_radius_m = iaat.utils.hexagon_outer_radius_given_inner_radius(
    inner_radius=feed_horn_inner_radius_m
)

feed_horn_inner_diameter_m = 2.0 * feed_horn_inner_radius_m
feed_horn_focal_length_m = focal_ratio_1 * feed_horn_inner_diameter_m

num = int(np.ceil(screen_radius_m / feed_horn_inner_radius_m))

feed_horn_spacing_m = 2.0 * feed_horn_inner_radius_m
big_feed_horn_center_grid = (
    optic_object_wavefronts.geometry.grid.hexagonal.init_from_spacing(
        spacing=feed_horn_spacing_m,
        fN=num,
    )
)

feed_horn_center_grid = {}
for key in big_feed_horn_center_grid:
    if np.linalg.norm(big_feed_horn_center_grid[key]) < screen_radius_m:
        feed_horn_center_grid[key] = big_feed_horn_center_grid[key]

feed_horn_voronoi_grid, feed_horn_voronoi_grid_edges = (
    optic_object_wavefronts.geometry.grid.hexagonal.init_voronoi_cells_from_centers(
        centers=feed_horn_center_grid,
        centers_spacing=feed_horn_spacing_m,
        return_edges=True,
    )
)

ORDER = 2
expected_scatter_spacing = feed_horn_outer_radius_m / (ORDER + 1)
_feed_horn_scatter_grid_prototype = (
    optic_object_wavefronts.geometry.grid.hexagonal.init_from_outer_radius(
        outer_radius=feed_horn_outer_radius_m - 0.5 * expected_scatter_spacing,
        fn=ORDER,
        ref=posixpath.join("foo", "inner"),
    )
)

feed_horn_scatter_grid_prototype = {}
PHI = np.pi / 6
for key in _feed_horn_scatter_grid_prototype:
    _x, _y, _ = _feed_horn_scatter_grid_prototype[key]
    ox = np.cos(PHI) * _x + np.sin(PHI) * _y
    oy = np.sin(PHI) * _x - np.cos(PHI) * _y

    _z = iaat.telescope.make_parabola_surface_height_m(
        distance_to_optical_axis_m=np.linalg.norm([_x, _y]),
        focal_length_m=feed_horn_focal_length_m,
    )
    feed_horn_scatter_grid_prototype[key] = np.array([ox, oy, _z])


fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])

for key in feed_horn_voronoi_grid:
    ax.plot(
        feed_horn_voronoi_grid[key][0],
        feed_horn_voronoi_grid[key][1],
        marker=".",
        color="black",
        markersize=0.8,
        linewidth=0.0,
    )

for feed_horn_key in feed_horn_center_grid:
    _edges = feed_horn_voronoi_grid_edges[feed_horn_key]
    for i in range(len(_edges)):
        start_key = _edges[i]
        stop_index = 0 if i == 5 else i + 1
        stop_key = _edges[stop_index]
        start_pos = feed_horn_voronoi_grid[start_key]
        stop_pos = feed_horn_voronoi_grid[stop_key]
        ax.plot(
            [start_pos[0], stop_pos[0]],
            [start_pos[1], stop_pos[1]],
            color="black",
            linewidth=0.2,
        )

for feed_horn_key in feed_horn_center_grid:
    for key in feed_horn_scatter_grid_prototype:
        total_x = (
            feed_horn_center_grid[feed_horn_key][0]
            + feed_horn_scatter_grid_prototype[key][0]
        )
        total_y = (
            feed_horn_center_grid[feed_horn_key][1]
            + feed_horn_scatter_grid_prototype[key][1]
        )
        ax.plot(total_x, total_y, marker=".", color="blue", linewidth=0.0)

for key in feed_horn_center_grid:
    ax.plot(
        feed_horn_center_grid[key][0],
        feed_horn_center_grid[key][1],
        marker=".",
        color="red",
        linewidth=0.0,
        markersize=0.8,
    )


ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_aspect("equal")
fig.savefig(os.path.join(out_dir, "feed_horn_mesh.jpg"))
sebplt.close(fig)
