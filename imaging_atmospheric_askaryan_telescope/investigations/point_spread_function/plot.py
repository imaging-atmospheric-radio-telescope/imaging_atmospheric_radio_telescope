import numpy as np
import sebastians_matplotlib_addons as sebplt

from ... import utils
from ... import plot as iaat_plot
from ... import camera as iaat_camera


def ax_add_feed_horn_hexagon(ax, x, y, feed_horn_area_m2, **kwargs):
    inner_radius_m = utils.inner_radius_of_hexagon(area=feed_horn_area_m2)
    outer_radius_m = 2.0 / np.sqrt(3.0) * inner_radius_m
    sebplt.ax_add_hexagon(
        ax=ax, x=x, y=y, r_outer=outer_radius_m, **kwargs, orientation_deg=30.0
    )


def ax_add_antenna_area_circle(ax, x, y, area_m2, **kwargs):
    r = np.sqrt(area_m2 / np.pi)
    sebplt.ax_add_circle(
        ax=ax,
        x=x,
        y=y,
        r=r,
        num_steps=101,
        **kwargs,
    )


def ax_add_wavelength_axis(ax, x, y, wavelength, **kwargs):
    w = wavelength
    ax.plot([x - w * 0.1, x + w * 1.1], [y, y], **kwargs)
    ax.plot([x, x], [y - w * 0.6, y + 0.6 * w], **kwargs)


def ax_add_wavelength_sine(ax, x, y, wavelength, **kwargs):
    w = wavelength
    _x = np.linspace(0.0, 1.0, 101)
    _y = np.sin(_x * 2.0 * np.pi)
    ax.plot(x + _x * w, y + _y * 0.5 * w, **kwargs)


def plot_camera(camera, energy_feed_horns_eV, path, feed_horn_mask=None):

    vmin, vmax = iaat_plot.log10_limits(energy_feed_horns_eV)

    fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.25, 0.65, 0.65])
    ax_cmap = sebplt.add_axes(fig=fig, span=[0.83, 0.25, 0.025, 0.65])
    norm = sebplt.matplotlib.colors.PowerNorm(
        vmin=1e-3 * np.max(energy_feed_horns_eV),
        vmax=np.max(energy_feed_horns_eV),
        gamma=1.0,
    )
    im = iaat_plot.ax_add_hexagonal_pixels(
        ax=ax,
        v=energy_feed_horns_eV,
        x=camera["feed_horn_positions_m"][:, 0],
        y=camera["feed_horn_positions_m"][:, 1],
        hexrotation=0,
        hex_inner_radius=camera["camera"]["feed_horn_inner_radius_m"],
        cmap="Blues",
        norm=norm,
        edgecolor="black",
        linewidth=0.1,
    )

    if feed_horn_mask is not None:
        for i in range(camera["num_feed_horns"]):
            if feed_horn_mask[i]:
                ax.plot(
                    camera["feed_horn_positions_m"][i, 0],
                    camera["feed_horn_positions_m"][i, 1],
                    marker="o",
                    color="red",
                )
    sebplt.ax_add_circle(
        ax=ax,
        x=0.0,
        y=0.0,
        r=camera["camera"]["outer_radius_m"],
        color="black",
        linewidth=0.2,
    )
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal")
    sebplt.plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel(r"energy / eV")

    ax_hist = sebplt.add_axes(fig=fig, span=[0.15, 0.05, 0.65, 0.12])
    bin_edges = np.geomspace(
        vmin, vmax, int(np.sqrt(len(energy_feed_horns_eV)))
    )
    hist = np.histogram(energy_feed_horns_eV, bins=bin_edges)[0]
    sebplt.ax_add_histogram(
        ax=ax_hist,
        bin_edges=bin_edges,
        bincounts=hist,
        draw_bin_walls=True,
    )
    ax_hist.set_xlim([vmin, vmax])
    ax_hist.set_ylim([0.5, len(energy_feed_horns_eV)])
    ax_hist.loglog()
    fig.savefig(path)
    sebplt.close(fig)


def plot_feed_horn_scatter_centers(camera, energy_feed_horns_scatter_eV, path):

    vmin, vmax = iaat_plot.log10_limits(energy_feed_horns_scatter_eV)

    scatpos = iaat_camera.get_camera_feed_horn_scatter_centers(camera=camera)

    fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.25, 0.65, 0.65])
    ax_cmap = sebplt.add_axes(fig=fig, span=[0.83, 0.25, 0.025, 0.65])
    _RRR = 1.05 * camera["camera"]["outer_radius_m"]
    _rrr = 0.5 * np.sqrt(camera["feed_horn_scatter_center_area_m2"])
    norm = sebplt.matplotlib.colors.PowerNorm(
        vmin=1e-3 * np.max(energy_feed_horns_scatter_eV),
        vmax=np.max(energy_feed_horns_scatter_eV),
        gamma=1.0,
    )
    patches = []
    for d in range(len(scatpos)):
        patches.append(
            sebplt.matplotlib.patches.RegularPolygon(
                (scatpos[d][0], scatpos[d][1]),
                numVertices=6,
                radius=_rrr,
                orientation=0.0,
            )
        )
    p = sebplt.matplotlib.collections.PatchCollection(patches, cmap="Blues")
    p.set_array(energy_feed_horns_scatter_eV)
    sebplt.plt.colorbar(p, cax=ax_cmap)
    ax_cmap.set_ylabel(r"energy / eV")
    iaat_camera.ax_add_camera_feed_horn_edges(
        ax=ax, camera=camera, color="black", alpha=0.33, linewidth=0.2
    )
    ax.add_collection(p)
    ax.set_xlim([-_RRR, _RRR])
    ax.set_ylim([-_RRR, _RRR])
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    ax_hist = sebplt.add_axes(fig=fig, span=[0.15, 0.05, 0.65, 0.12])
    bin_edges_eV = np.geomspace(
        vmin, vmax, int(np.sqrt(len(energy_feed_horns_scatter_eV)))
    )
    hist = np.histogram(energy_feed_horns_scatter_eV, bins=bin_edges_eV)[0]
    sebplt.ax_add_histogram(
        ax=ax_hist,
        bin_edges=bin_edges_eV,
        bincounts=hist,
        draw_bin_walls=True,
    )
    ax_hist.set_xlim([vmin, vmax])
    ax_hist.set_ylim([0.5, len(energy_feed_horns_scatter_eV)])
    ax_hist.loglog()
    ax_hist.set_xlabel("energy / eV")
    ax_hist.set_ylabel("num. channels / 1")

    fig.savefig(path)
    sebplt.close(fig)
