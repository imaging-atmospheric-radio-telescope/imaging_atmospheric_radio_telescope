from . import utils

import collections
import numpy as np
import optic_object_wavefronts as oow
import thin_lens
import copy


def _make_feed_horn_center_grid(
    screen_radius_m,
    feed_horn_inner_radius_m,
):
    fN = int(np.ceil(screen_radius_m / feed_horn_inner_radius_m))
    feed_horn_spacing_m = 2.0 * feed_horn_inner_radius_m
    big_feed_horn_center_grid = oow.geometry.grid.hexagonal.init_from_spacing(
        spacing=feed_horn_spacing_m,
        fN=fN,
    )
    return _cut_inside_radius(
        grid=big_feed_horn_center_grid,
        radius=screen_radius_m,
    )


def _cut_inside_radius(grid, radius):
    out = collections.OrderedDict()
    for key in grid:
        if np.linalg.norm([grid[key][0], grid[key][1]]) < radius:
            out[key] = grid[key]
    return out


def _rotate_grid_in_xy_plane(grid, angle_rad):
    out = collections.OrderedDict()
    cosA = np.cos(angle_rad)
    sinA = np.sin(angle_rad)
    for key in grid:
        _x, _y, _z = grid[key]
        ox = cosA * _x + sinA * _y
        oy = sinA * _x - cosA * _y
        out[key] = np.array([ox, oy, _z])
    return out


def _grid_add_vector(grid, vector):
    vector = np.array(vector)
    out = collections.OrderedDict()
    for key in grid:
        out[key] = grid[key] + vector
    return out


def _make_feed_horn_scatter_centers_only_xy(
    feed_horn_inner_radius_m,
    feed_horn_oversampling_order,
):
    feed_horn_outer_radius_m = utils.hexagon_outer_radius_given_inner_radius(
        inner_radius=feed_horn_inner_radius_m
    )

    expected_scatter_spacing = (2 * feed_horn_outer_radius_m) / (
        feed_horn_oversampling_order + 2
    )
    return oow.geometry.grid.hexagonal.init_from_outer_radius(
        outer_radius=feed_horn_outer_radius_m - 0.5 * expected_scatter_spacing,
        fn=feed_horn_oversampling_order,
        ref="scatter",
    )


def _make_feed_horn_focal_length_m(
    feed_horn_inner_radius_m, feed_horn_focal_ratio_1
):
    feed_horn_inner_diameter_m = 2.0 * feed_horn_inner_radius_m
    return feed_horn_focal_ratio_1 * feed_horn_inner_diameter_m


def _make_feed_horn_scatter_centers(
    feed_horn_inner_radius_m,
    feed_horn_oversampling_order,
):
    scatter_grid = _make_feed_horn_scatter_centers_only_xy(
        feed_horn_inner_radius_m=feed_horn_inner_radius_m,
        feed_horn_oversampling_order=feed_horn_oversampling_order,
    )
    scatter_grid = _rotate_grid_in_xy_plane(
        grid=scatter_grid,
        angle_rad=np.pi / 6,
    )
    return scatter_grid


def _make_feed_horn_grid_and_edges(
    feed_horn_positions_grid,
    feed_horn_inner_radius_m,
):
    feed_horn_spacing_m = 2.0 * feed_horn_inner_radius_m
    feed_horn_voronoi_grid, feed_horn_voronoi_grid_edges = (
        oow.geometry.grid.hexagonal.init_voronoi_cells_from_centers(
            centers=feed_horn_positions_grid,
            centers_spacing=feed_horn_spacing_m,
            return_edges=True,
        )
    )
    return feed_horn_voronoi_grid, feed_horn_voronoi_grid_edges


def _flatten_grid(grid):
    out = []
    mapping = {}
    for i, key in enumerate(grid):
        out.append(grid[key])
        mapping[key] = i
    return np.array(out), mapping


def make_camera(
    sensor_outer_radius_m,
    sensor_distance_m,
    feed_horn_inner_radius_m,
    feed_horn_transmission,
    feed_horn_oversampling_order,
):
    assert sensor_outer_radius_m > 0.0
    assert sensor_distance_m > 0.0
    assert feed_horn_inner_radius_m > 0.0
    assert feed_horn_oversampling_order >= 1
    assert 0.0 <= feed_horn_transmission <= 1.0

    c = {}
    c["__type__"] = "camera"
    c["camera"] = {}
    c["camera"]["outer_radius_m"] = sensor_outer_radius_m
    c["camera"]["outer_diameter_m"] = 2 * sensor_outer_radius_m
    c["camera"]["feed_horn_inner_radius_m"] = feed_horn_inner_radius_m
    c["camera"]["feed_horn_oversampling_order"] = feed_horn_oversampling_order

    feed_horn_positions_grid = _make_feed_horn_center_grid(
        screen_radius_m=sensor_outer_radius_m,
        feed_horn_inner_radius_m=feed_horn_inner_radius_m,
    )
    feed_horn_scatter_grid_prototype = _make_feed_horn_scatter_centers(
        feed_horn_inner_radius_m=feed_horn_inner_radius_m,
        feed_horn_oversampling_order=feed_horn_oversampling_order,
    )
    feed_horn_voronoi_grid, feed_horn_voronoi_grid_edges = (
        _make_feed_horn_grid_and_edges(
            feed_horn_positions_grid=feed_horn_positions_grid,
            feed_horn_inner_radius_m=feed_horn_inner_radius_m,
        )
    )
    feed_horn_positions_grid = _grid_add_vector(
        grid=feed_horn_positions_grid,
        vector=[0, 0, sensor_distance_m],
    )

    c["feed_horn_positions_m"], _center_map = _flatten_grid(
        grid=feed_horn_positions_grid
    )
    c["feed_horn_transmission"] = feed_horn_transmission
    c["num_feed_horns"] = len(c["feed_horn_positions_m"])
    c["feed_horn_area_m2"] = utils.area_of_hexagon(
        inner_radius=c["camera"]["feed_horn_inner_radius_m"]
    )

    c["feed_horn_relative_scatter_center_positions_m"], _ = _flatten_grid(
        grid=feed_horn_scatter_grid_prototype
    )
    c["num_scatter_centers_per_feed_horn"] = len(
        c["feed_horn_relative_scatter_center_positions_m"]
    )
    c["feed_horn_scatter_center_area_m2"] = (
        c["feed_horn_area_m2"] / c["num_scatter_centers_per_feed_horn"]
    )

    c["camera"]["feed_horn_edge_vertices_m"], _voronoi_map = _flatten_grid(
        grid=feed_horn_voronoi_grid
    )

    c["camera"]["feed_horn_edge_mapping"] = []
    for feed_horn_key in feed_horn_positions_grid:
        gg = []
        for edge_vertex_key in feed_horn_voronoi_grid_edges[feed_horn_key]:
            gg.append(_voronoi_map[edge_vertex_key])
        c["camera"]["feed_horn_edge_mapping"].append(gg)
    c["camera"]["feed_horn_edge_mapping"] = np.array(
        c["camera"]["feed_horn_edge_mapping"]
    )

    c["sensor_distance_m"] = sensor_distance_m
    return c


def ax_add_camera(ax, camera, **kwargs):
    ax_add_camera_feed_horn_edges(
        ax=ax, camera=camera, linewidth=1.0, alpha=0.2, **kwargs
    )
    ax_add_camera_feed_horn_scatter_centers(
        ax=ax, camera=camera, marker="o", alpha=0.33, **kwargs
    )
    ax_add_camera_feed_horn_centers(ax=ax, camera=camera, marker="+", **kwargs)


def ax_add_camera_feed_horn_centers(ax, camera, **kwargs):
    for i in range(len(camera["feed_horn_positions_m"])):
        ax.plot(
            camera["feed_horn_positions_m"][i][0],
            camera["feed_horn_positions_m"][i][1],
            **kwargs,
        )


def ax_add_camera_feed_horn_edges(ax, camera, **kwargs):
    for i in range(len(camera["feed_horn_positions_m"])):
        _edges = camera["camera"]["feed_horn_edge_mapping"][i]
        for j in range(len(_edges)):
            start_key = _edges[j]
            stop_index = 0 if j == 5 else j + 1
            stop_key = _edges[stop_index]
            start_pos = camera["camera"]["feed_horn_edge_vertices_m"][
                start_key
            ]
            stop_pos = camera["camera"]["feed_horn_edge_vertices_m"][stop_key]
            ax.plot(
                [start_pos[0], stop_pos[0]],
                [start_pos[1], stop_pos[1]],
                **kwargs,
            )


def ax_add_camera_feed_horn_scatter_centers(ax, camera, **kwargs):
    poss = get_camera_feed_horn_scatter_centers(camera=camera)
    for pos in poss:
        ax.plot(pos[0], pos[1], **kwargs)


def get_camera_feed_horn_scatter_centers(camera):
    positions_m = []
    for i in range(camera["num_feed_horns"]):
        for j in range(camera["num_scatter_centers_per_feed_horn"]):
            position_m = (
                camera["feed_horn_positions_m"][i]
                + camera["feed_horn_relative_scatter_center_positions_m"][j]
            )
            positions_m.append(position_m)
    return np.array(positions_m)
