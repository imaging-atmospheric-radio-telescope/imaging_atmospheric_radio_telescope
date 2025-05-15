#!/usr/bin/env python3
import argparse
import sebastians_matplotlib_addons as sebplt

import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import spherical_coordinates
import numpy as np
import json_utils
import os
import scipy.linalg

telescope_key = "crome"
work_dir = f"explore_point_spread_function_{telescope_key:s}"

if not os.path.exists(work_dir):
    iaat.run.init(
        work_dir=work_dir,
        site_key="namibia",
        telescope_key=telescope_key,
    )

_askaryan = iaat.run.from_config(work_dir=work_dir)
telescope = _askaryan["telescope"]
timing = _askaryan["timing"]
site = _askaryan["site"]

random_seed = 1405

lnb_start_Hz, lnb_stop_Hz = iaat.lownoiseblock.input_frequency_start_stop_Hz(
    lnb=telescope["lnb"]
)
lnb_input_frequency_Hz = np.mean([lnb_start_Hz, lnb_stop_Hz])


# HEAD ON
# -------
source_config = iaat.production.radio_from_plane_wave.make_config()

s1 = iaat.calibration_source.plane_wave_in_far_field.make_config()
s1["geometry"]["azimuth_rad"] = np.deg2rad(0.0)
s1["geometry"]["zenith_rad"] = np.deg2rad(0.0)
s1["power"]["power_of_isotrop_and_point_like_emitter_W"] = 2e-1
s1["sine_wave"]["emission_frequency_Hz"] = lnb_input_frequency_Hz * 1.01

s2 = iaat.calibration_source.plane_wave_in_far_field.make_config()
s2["geometry"]["azimuth_rad"] = np.deg2rad(50)
s2["geometry"]["zenith_rad"] = (
    0.45
    * telescope["sensor"]["camera"]["outer_radius_m"]
    / telescope["mirror"]["focal_length_m"]
)
s2["power"]["power_of_isotrop_and_point_like_emitter_W"] = 4e-1
s2["sine_wave"]["emission_frequency_Hz"] = lnb_input_frequency_Hz * 0.99

source_config["plane_waves"] = {}
source_config["plane_waves"]["first"] = s1
source_config["plane_waves"]["second"] = s2

scenario_dir = os.path.join(work_dir, "response")


region_of_interest_rad = (
    6
    * np.sqrt(telescope["sensor"]["feed_horn_area_m2"])
    / telescope["mirror"]["focal_length_m"]
)


iaat.investigations.point_spread_function.plane_wave_response.make_PlaneWaveResponse(
    out_dir=scenario_dir,
    random_seed=random_seed,
    telescope=telescope,
    site=site,
    timing=timing,
    source_config=source_config,
    region_of_interest_rad=region_of_interest_rad,
    region_of_interest_num_bins=21,
    save_feed_horns_scatter_electric_fields=True,
    save_roi_electric_fields=True,
)
response = iaat.investigations.point_spread_function.plane_wave_response.PlaneWaveResponse(
    path=scenario_dir
)

I_energy_eV = response.energy_feed_horns / iaat.signal.ELECTRON_VOLT_J

iaat.investigations.point_spread_function.plot.plot_camera(
    camera=telescope["sensor"],
    energy_feed_horns_eV=I_energy_eV,
    path=os.path.join(scenario_dir, "camera.jpg"),
)

fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.65, 0.65])
iaat.camera.ax_add_camera(ax=ax, camera=telescope["sensor"], color="black")
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_aspect("equal")
fig.savefig(os.path.join(work_dir, "feed_horn_mesh.jpg"))
sebplt.close(fig)


R_airy_m = iaat.telescope.calculate_airy_disk_radius_in_focal_plane(
    telescope=telescope
)

E_feed_horns = iaat.time_series.read(
    os.path.join(
        work_dir,
        "response",
        "camera",
        "feed_horns",
        "scatter.electric_fields.tar",
    )
)
Ene_feed_horn_scatters_J = iaat.electric_fields.integrate_power_over_time(
    electric_fields=E_feed_horns,
    channel_effective_area_m2=telescope["sensor"][
        "feed_horn_scatter_center_area_m2"
    ],
)
Ene_pixel_J = np.zeros(telescope["sensor"]["num_feed_horns"])
for iii in range(telescope["sensor"]["num_feed_horns"]):
    iii_start = iii * telescope["sensor"]["num_scatter_centers_per_feed_horn"]
    iii_stop = (iii + 1) * telescope["sensor"][
        "num_scatter_centers_per_feed_horn"
    ]
    Ene_pixel_J[iii] = np.sum(Ene_feed_horn_scatters_J[iii_start:iii_stop])
Ene_pixel_eV = Ene_pixel_J / iaat.signal.ELECTRON_VOLT_J


iaat.investigations.point_spread_function.plot.plot_camera(
    camera=telescope["sensor"],
    energy_feed_horns_eV=Ene_pixel_eV,
    path=os.path.join(scenario_dir, "camera_from_fine.jpg"),
)

Ene_feed_horn_scatters_eV = (
    Ene_feed_horn_scatters_J / iaat.signal.ELECTRON_VOLT_J
)

iaat.investigations.point_spread_function.plot.plot_feed_horn_scatter_centers(
    camera=telescope["sensor"],
    energy_feed_horns_scatter_eV=Ene_feed_horn_scatters_eV,
    path=os.path.join(scenario_dir, "camera_fine.jpg"),
)


for key in response.region_of_interest_keys:
    feed_horn_mask = iaat.investigations.point_spread_function.plane_wave_response.mask_feed_horns(
        feed_horn_positions_m=telescope["sensor"]["feed_horn_positions_m"],
        containment_radius_m=2.0 * R_airy_m,
        azimuth_rad=response.source_config["plane_waves"][key]["geometry"][
            "azimuth_rad"
        ],
        zenith_rad=response.source_config["plane_waves"][key]["geometry"][
            "zenith_rad"
        ],
    )

    iaat.investigations.point_spread_function.plot.plot_camera(
        camera=telescope["sensor"],
        energy_feed_horns_eV=np.ones(telescope["sensor"]["num_feed_horns"]),
        path=os.path.join(scenario_dir, f"{key:s}_mask.jpg"),
        feed_horn_mask=feed_horn_mask,
    )

    Ene_expected_to_be_collected_by_mirror_J = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
        config=response.source_config["plane_waves"][key],
        area_m2=telescope["mirror"]["area_m2"],
    )

    Ene_mirror_J = iaat.electric_fields.integrate_power_over_time(
        electric_fields=response.E_mirror,
        channel_effective_area_m2=telescope["mirror"][
            "scatter_center_area_m2"
        ],
    )
    Ene_mirror_J = np.sum(Ene_mirror_J)

    Ene_camera_J = iaat.electric_fields.integrate_power_over_time(
        electric_fields=response.E_feed_horns,
        channel_effective_area_m2=response.sensor["feed_horn_area_m2"],
    )

    Ene_expected_to_be_collected_by_mirror_eV = (
        Ene_expected_to_be_collected_by_mirror_J / iaat.signal.ELECTRON_VOLT_J
    )
    Ene_mirror_eV = Ene_mirror_J / iaat.signal.ELECTRON_VOLT_J
    Ene_camera_eV = Ene_camera_J / iaat.signal.ELECTRON_VOLT_J

    print(f"__source__: {key:s}")
    print(
        f"Expected in ROI:{Ene_expected_to_be_collected_by_mirror_eV: 5.2f}eV"
    )
    print(f"Mirror         :{Ene_mirror_eV: 5.2f}eV")
    # print(f"Fine sum       :{np.sum(Ene_pixel_eV): 5.2f}eV")
    print(f"Fine ROI       :{np.sum(Ene_pixel_eV[feed_horn_mask]): 5.2f}eV")
    # print(f"Camera sum     :{np.sum(Ene_camera_eV): 5.2f}eV")
    print(f"Camera ROI     :{np.sum(Ene_camera_eV[feed_horn_mask]): 5.2f}eV")

    bx, by, Ene_img_J = response.energy_roi(key)
    ana = iaat.investigations.point_spread_function.power_image_analysis.analyse_image(
        x_bin_edges_m=bx,
        y_bin_edges_m=by,
        image=Ene_img_J,
        containment_quantile=0.8,
    )

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
    """
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
    """
    ax.plot(ana["argmax_x_m"], ana["argmax_y_m"], marker="o", color="red")
    sebplt.ax_add_circle(
        ax=ax,
        x=ana["argmax_x_m"],
        y=ana["argmax_y_m"],
        r=ana["radius_quantile_m"],
        linestyle="--",
        color="red",
    )

    """
    sebplt.ax_add_circle(
        ax=ax,
        x=0.0,
        y=0.0,
        r=telescope["sensor"]["camera"]["outer_radius_m"],
        color="black",
    )
    """
    iaat.camera.ax_add_camera_feed_horn_edges(
        ax=ax,
        camera=telescope["sensor"],
        color="black",
        linewidth=0.5,
    )
    iaat.camera.ax_add_camera_feed_horn_scatter_centers(
        ax=ax,
        camera=telescope["sensor"],
        color="black",
        marker=".",
        alpha=0.33,
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

    """
    E_feed_horns = response.E_feed_horns
    for channel in range(E_feed_horns.num_channels):
        factor, vector = (
            iaat.investigations.point_spread_function.polarization_analysis.analyse_linear_polarization_over_time(
                E_field_vs_time=E_feed_horns[channel]
            )
        )
        print("camera", channel, "polarization", factor, vector)

    E_mirror = response.E_mirror
    for channel in range(E_mirror.num_channels):
        factor, vector = (
            iaat.investigations.point_spread_function.polarization_analysis.analyse_linear_polarization_over_time(
                E_field_vs_time=E_mirror[channel]
            )
        )
        print("mirror", channel, "polarization", factor, vector)
    """
