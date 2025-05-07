#!/usr/bin/env python3
import argparse
import sebastians_matplotlib_addons as sebplt

import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import spherical_coordinates
import numpy as np
import json_utils
import os

work_dir = "explore_point_spread_function"

if not os.path.exists(work_dir):
    iaat.run.init(
        work_dir=work_dir,
        site_key="namibia",
        telescope_key="large_size_telescope",
    )

_askaryan = iaat.run.from_config(work_dir=work_dir)
telescope = _askaryan["telescope"]
timing = _askaryan["timing"]
site = _askaryan["site"]

random_seed = 1405


# HEAD ON
# -------
source_config = iaat.production.radio_from_plane_wave.make_config()
s1 = source_config["plane_waves"]["first"]
s1["geometry"]["azimuth_rad"] = np.deg2rad(220)
s1["geometry"]["zenith_rad"] = np.deg2rad(1.8)
s1["geometry"][
    "distance_to_plane_defining_time_zero_m"
] = iaat.corsika.TOP_OF_ATMOSPHERE_ALTITUDE_M
s1["power"]["power_of_isotrop_and_point_like_emitter_W"] = 2e-1
s1["power"]["distance_to_isotrop_and_point_like_emitter_m"] = 100e3
s1["sine_wave"]["emission_frequency_Hz"] = 11.1e9
s1["sine_wave"]["emission_duration_s"] = 5e-9
s1["sine_wave"]["emission_ramp_up_duration_s"] = 1e-9
s1["sine_wave"]["emission_ramp_down_duration_s"] = 1e-9

scenario_dir = os.path.join(work_dir, "response")


iaat.investigations.point_spread_function.make_PlaneWaveResponse(
    out_dir=scenario_dir,
    random_seed=random_seed,
    telescope=telescope,
    site=site,
    timing=timing,
    source_config=source_config,
    region_of_interest_rad=np.deg2rad(0.5),
    region_of_interest_num_bins=21,
)
response = iaat.investigations.point_spread_function.PlaneWaveResponse(
    path=scenario_dir
)

I_energy_eV = response.Image_energy / iaat.signal.ELECTRON_VOLT_J

fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
ax_cmap = sebplt.add_axes(fig=fig, span=[0.83, 0.15, 0.025, 0.65])
norm = sebplt.matplotlib.colors.PowerNorm(
    vmin=1e-2 * np.max(I_energy_eV),
    vmax=np.max(I_energy_eV),
    gamma=1 / 2.0,
)
im = iaat_plot.ax_add_hexagonal_pixels(
    ax=ax,
    v=I_energy_eV,
    x=response.sensor["feed_horn_positions_m"][:, 0],
    y=response.sensor["feed_horn_positions_m"][:, 1],
    cmap="viridis",
    hexrotation=0,
    norm=norm,
)
sebplt.ax_add_circle(
    ax=ax,
    x=0.0,
    y=0.0,
    r=response.sensor["camera"]["outer_radius_m"],
    color="black",
)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_aspect("equal")
sebplt.plt.colorbar(im, cax=ax_cmap)
ax_cmap.set_ylabel(r"Energy / eV")
fig.savefig(os.path.join(scenario_dir, f"camera.jpg"))
sebplt.close(fig)


for key in response.region_of_interest_keys:

    bx, by, Ene_img_J = response.Image_energy_roi(key)
    ana = iaat.investigations.point_spread_function.analyse_image(Ene_img_J)

    Ene_img_eV = Ene_img_J / iaat.signal.ELECTRON_VOLT_J

    fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 1.5})
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
    ax_cmap = sebplt.add_axes(fig=fig, span=[0.83, 0.15, 0.025, 0.65])
    norm = sebplt.matplotlib.colors.PowerNorm(
        vmin=1e-3 * np.max(Ene_img_eV),
        vmax=np.max(Ene_img_eV),
        gamma=1 / 2.0,
    )
    im = ax.pcolormesh(
        bx,
        by,
        Ene_img_eV.T,
        cmap="Blues",
        norm=norm,
    )
    iaat.investigations.point_spread_function.plot.ax_add_feed_horn_hexagon(
        ax=ax,
        x=bx[17],
        y=by[17],
        feed_horn_area_m2=telescope["sensor"]["feed_horn_area_m2"],
        color="black",
        linewidth=0.7,
    )
    iaat.investigations.point_spread_function.plot.ax_add_antenna_area_circle(
        ax=ax,
        x=bx[17],
        y=by[17],
        area_m2=iaat.signal.calculate_antenna_effective_area(
            wavelength=iaat.signal.frequency_to_wavelength(11.1e9), gain=1.0
        ),
        color="black",
        linewidth=0.7,
    )
    w = iaat.signal.frequency_to_wavelength(
        response.source_config["plane_waves"][key]["sine_wave"][
            "emission_frequency_Hz"
        ]
    )
    iaat.investigations.point_spread_function.plot.ax_add_wavelength_axis(
        ax=ax, x=bx[17], y=by[4], wavelength=w, color="gray", linewidth=0.5
    )
    iaat.investigations.point_spread_function.plot.ax_add_wavelength_sine(
        ax=ax, x=bx[17], y=by[4], wavelength=w, color="black", linewidth=0.5
    )
    sebplt.ax_add_circle(
        ax=ax,
        x=0.0,
        y=0.0,
        r=telescope["sensor"]["camera"]["outer_radius_m"],
        color="black",
    )
    ax.set_xlim([min(bx), max(bx)])
    ax.set_ylim([min(by), max(by)])
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal")
    sebplt.plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel(r"Energy / eV")

    fig.savefig(os.path.join(scenario_dir, f"{key:s}.jpg"))
    sebplt.close(fig)
