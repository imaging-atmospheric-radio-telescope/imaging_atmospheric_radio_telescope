import argparse
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import json_utils


parser = argparse.ArgumentParser(
    prog="plot_wave_package.py",
    description=("Plot a the wave package."),
)
parser.add_argument(
    "--out_dir",
    metavar="OUT_DIR",
    default="wave_package",
    type=str,
    help="Path to write figures to.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


args = parser.parse_args()
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

time_slice_duration_s = 1 / (65e9)
exposure_time_s = 7e-9
N = int(exposure_time_s / time_slice_duration_s)

config = iaat.calibration_source.plane_wave_in_far_field.make_config()

moon_to_earth_m = 385_000_000
iss_altitude_m = 422_000

config["power"] = {
    "power_of_isotrop_and_point_like_emitter_W": 1.0,
    "distance_to_isotrop_and_point_like_emitter_m": iss_altitude_m,
}

config["sine_wave"]["emission_frequency_Hz"] = 10.32e9

unit_area_m2 = 1.0
areal_energy_density_J_per_m2 = iaat.calibration_source.plane_wave_in_far_field.calculate_total_energy_from_config(
    config, area_m2=unit_area_m2
)
areal_power_density_at_main_part_W_per_m2 = config["power"][
    "power_of_isotrop_and_point_like_emitter_W"
] / (
    4
    * np.pi
    * config["power"]["distance_to_isotrop_and_point_like_emitter_m"] ** 2
)

with open(os.path.join(out_dir, "wave_package.json"), "wt") as f:
    f.write(
        json_utils.dumps(
            {
                "config": config,
                "areal_energy_density_J_per_m2": areal_energy_density_J_per_m2,
                "areal_power_density_at_main_part_W_per_m2": areal_power_density_at_main_part_W_per_m2,
            },
            indent=4,
        )
    )

ppp = [[0, 0, 0]]

E_plane_wave = iaat.calibration_source.plane_wave_in_far_field.plane_wave_in_far_field(
    geometry_setup=iaat.calibration_source.plane_wave_in_far_field.make_geometry_setup(
        antenna_position_vectors_in_asl_frame_m=ppp,
        **config["geometry"],
    ),
    power_setup=iaat.calibration_source.plane_wave_in_far_field.make_power_setup(
        **config["power"],
    ),
    sine_wave=config["sine_wave"],
    time_slice_duration_s=time_slice_duration_s,
)

t_offset_ns = 1
t_ns = (
    1e9 * E_plane_wave.make_time_bin_centers(global_time=False) - t_offset_ns
)
Ax_uV_per_m = 1e6 * E_plane_wave[0, :, 0]
max_A_uV_per_m = np.max(Ax_uV_per_m)


KEYS = {
    "normal": {"cols": 1920, "xs": 0.15, "xw": 0.8},
    "wide": {"cols": 3840, "xs": 0.075, "xw": 0.9},
}
for key in KEYS:
    fig = sebplt.figure(
        {"rows": 1080, "cols": KEYS[key]["cols"], "fontsize": 2.0}
    )
    ax = sebplt.add_axes(
        fig=fig, span=[KEYS[key]["xs"], 0.2, KEYS[key]["xw"], 0.75]
    )
    xo = np.array([0, 1, 2, 7, 8, 9]) - t_offset_ns
    yo = np.array([0, 0, max_A_uV_per_m, max_A_uV_per_m, 0, 0])
    ax.fill(xo, yo, color="lightblue", alpha=0.7)
    ax.fill(xo, -yo, color="lightblue", alpha=0.7)
    ax.plot(t_ns, Ax_uV_per_m, color="black", linewidth=0.5)
    ax.plot(
        t_ns,
        Ax_uV_per_m,
        color="black",
        linewidth=0.0,
        marker="o",
        markersize=0.6,
    )
    ax.set_xlabel("time / ns")
    ax.set_xlim([min(t_ns) + 0.75, max(t_ns) - 0.75])
    ax.set_ylabel(r"electric field / $\mu$Vm$^{-1}$")
    fig.savefig(os.path.join(out_dir, f"wave_package_{key:s}.jpg"))
    sebplt.close(fig)
