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
telescope_full_camera = _askaryan["telescope"]
timing = _askaryan["timing"]
site = _askaryan["site"]

random_seed = 1405


# HEAD ON
# -------
scenario_key = "head_on"
scenario_dir = os.path.join(work_dir, "plane_wave", scenario_key)

source_config = iaat.production.radio_from_plane_wave.make_config()
source_config["geometry"]["azimuth_rad"] = np.deg2rad(0)
source_config["geometry"]["zenith_rad"] = np.deg2rad(2.1)
source_config["geometry"][
    "distance_to_plane_defining_time_zero_m"
] = iaat.corsika.TOP_OF_ATMOSPHERE_ALTITUDE_M
source_config["power"]["power_of_isotrop_and_point_like_emitter_W"] = 2e-1
source_config["power"]["distance_to_isotrop_and_point_like_emitter_m"] = 100e3
source_config["sine_wave"]["emission_frequency_Hz"] = 11.1e9
source_config["sine_wave"]["emission_duration_s"] = 5e-9
source_config["sine_wave"]["emission_ramp_up_duration_s"] = 1e-9
source_config["sine_wave"]["emission_ramp_down_duration_s"] = 1e-9


telescope_roi_camera = iaat.investigations.point_spread_function.make_telescope_like_other_but_with_region_of_interest_camera(
    source_azimuth_rad=source_config["geometry"]["azimuth_rad"],
    source_zenith_rad=source_config["geometry"]["zenith_rad"],
    region_of_interest_rad=np.deg2rad(0.5),
    num_bins=42,
    other_telescope=telescope_full_camera,
)

roi_dir = os.path.join(scenario_dir, "region_of_interest")
iaat.production.simulate_telescope_response(
    out_dir=roi_dir,
    source_config=source_config,
    site=site,
    telescope=telescope_roi_camera,
    timing=timing,
    thermal_noise_random_seed=random_seed + 1,
    readout_random_seed=random_seed + 2,
    stop_after_section="feed_horns",
)


E_roi = iaat.time_series.read(
    os.path.join(roi_dir, "feed_horns", "electric_fields.tar")
)
E_roi_magnitude_V_per_m = E_roi.norm_components()
P_roi_W = iaat.signal.calculate_antenna_power_W(
    effective_area_m2=telescope_roi_camera["sensor"]["feed_horn_area_m2"],
    electric_field_V_per_m=E_roi_magnitude_V_per_m[:],
)
Ene_roi_J = np.sum(P_roi_W, axis=1) * E_roi.time_slice_duration_s
Ene_roi_J = Ene_roi_J.reshape(
    (
        len(
            telescope_roi_camera["sensor"]["region_of_interest"][
                "x_bin_edges_m"
            ]
        )
        - 1,
        len(
            telescope_roi_camera["sensor"]["region_of_interest"][
                "y_bin_edges_m"
            ]
        )
        - 1,
    )
)
Ene_roi_eV = Ene_roi_J / iaat.signal.ELECTRON_VOLT_J


fig = sebplt.figure(style={"rows": 1920, "cols": 1920, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
ax_cmap = sebplt.add_axes(fig=fig, span=[0.83, 0.15, 0.025, 0.65])
norm = sebplt.matplotlib.colors.PowerNorm(
    vmin=1e-3 * np.max(Ene_roi_eV),
    vmax=np.max(Ene_roi_eV),
    gamma=1 / 2.0,
)
# ax.set_title(title, fontsize="small")
im = ax.pcolormesh(
    telescope_roi_camera["sensor"]["region_of_interest"]["x_bin_edges_m"],
    telescope_roi_camera["sensor"]["region_of_interest"]["y_bin_edges_m"],
    Ene_roi_eV.T,
    cmap="Blues",
    norm=norm,
)
iaat.investigations.point_spread_function.plot.ax_add_feed_horn_hexagon(
    ax=ax,
    x=telescope_roi_camera["sensor"]["region_of_interest"]["x_bin_edges_m"][
        17
    ],
    y=telescope_roi_camera["sensor"]["region_of_interest"]["y_bin_edges_m"][
        17
    ],
    feed_horn_area_m2=telescope_full_camera["sensor"]["feed_horn_area_m2"],
    color="black",
    linewidth=0.7,
)
iaat.investigations.point_spread_function.plot.ax_add_antenna_area_circle(
    ax=ax,
    x=telescope_roi_camera["sensor"]["region_of_interest"]["x_bin_edges_m"][
        17
    ],
    y=telescope_roi_camera["sensor"]["region_of_interest"]["y_bin_edges_m"][
        17
    ],
    area_m2=iaat.signal.calculate_antenna_effective_area(
        wavelength=iaat.signal.frequency_to_wavelength(11.1e9), gain=1.0
    ),
    color="black",
    linewidth=0.7,
)
w = iaat.signal.frequency_to_wavelength(
    source_config["sine_wave"]["emission_frequency_Hz"]
)
_x = telescope_roi_camera["sensor"]["region_of_interest"]["x_bin_edges_m"][17]
_y = telescope_roi_camera["sensor"]["region_of_interest"]["y_bin_edges_m"][4]
iaat.investigations.point_spread_function.plot.ax_add_wavelength_axis(
    ax=ax, x=_x, y=_y, wavelength=w, color="gray", linewidth=0.5
)
iaat.investigations.point_spread_function.plot.ax_add_wavelength_sine(
    ax=ax, x=_x, y=_y, wavelength=w, color="black", linewidth=0.5
)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_aspect("equal")
sebplt.plt.colorbar(im, cax=ax_cmap)
ax_cmap.set_ylabel(r"Energy / eV")

fig.savefig(os.path.join(scenario_dir, "region_of_interest.jpg"))
sebplt.close(fig)
