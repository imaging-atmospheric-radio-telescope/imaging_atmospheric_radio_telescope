import numpy as np
import sebastians_matplotlib_addons as sebplt

from ... import utils


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
