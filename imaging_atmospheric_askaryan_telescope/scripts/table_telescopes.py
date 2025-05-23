#!/usr/bin/env python3
import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import os
import io
import thin_lens


psf_dir = "2025-05-22_psf"
config = iaat.investigations.point_spread_function.utils.read_config(psf_dir)

OBJECT_DISTANCE_M = 10e3

ts = {}
for telescope_key in config["telescopes"]:
    sensor_distance_m = thin_lens.compute_image_distance_for_object_distance(
        object_distance=OBJECT_DISTANCE_M,
        focal_length=config["telescopes"][telescope_key]["mirror"][
            "focal_length_m"
        ],
    )

    t, _, _ = (
        iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
            work_dir=psf_dir,
            config=config,
            telescope_key=telescope_key,
            sensor_distance_m=sensor_distance_m,
        )
    )
    ts[telescope_key] = t


def comment_row(c):
    return [c, "", "", ""]


def tab_to_latex(tab):
    ss = ""
    for row in tab:
        for ii in range(len(row)):
            item = row[ii]
            ss += item
            if ii + 1 < len(row):
                ss += " & "
            else:
                ss += r" \\"
        ss += "\n"
    return ss


s = ""
s += r"       &        CROME &       Medium &        Large \\"
s += "\n"
s += r"symbol & $\mathbf{C}$ & $\mathbf{M}$ & $\mathbf{L}$ \\"
s += "\n"
s += "$D$/m  &"

# OPTICS Mirror
# =============

tab = []

tab.append(comment_row("outer diameter"))
row = []
row.append(r"$D$\,/\,m")
for key in ts:
    D = 2 * ts[key]["mirror"]["outer_radius_m"]
    row.append(f"{D:.2f}")
tab.append(row)

tab.append(comment_row("focal-length"))
row = []
row.append(r"$f$\,/\,m")
for key in ts:
    f = ts[key]["mirror"]["focal_length_m"]
    row.append(f"{f:.2f}")
tab.append(row)

tab.append(comment_row("focal-ratio"))
row = []
row.append(r"$f\,/\,D$")
for key in ts:
    f = ts[key]["mirror"]["focal_length_m"]
    D = 2 * ts[key]["mirror"]["outer_radius_m"]
    row.append(f"{f/D:.2f}")
tab.append(row)

tab.append(comment_row("inner diameter"))
row = []
row.append(r"$D_\text{inner}$\,/\,m")
for key in ts:
    Di = 2 * ts[key]["mirror"]["inner_radius_m"]
    row.append(f"{Di:.2f}")
tab.append(row)

tab.append(comment_row("area"))
row = []
row.append(r"$A_\text{mirror}$\,/\,m$^{2}$")
for key in ts:
    Am = ts[key]["mirror"]["area_m2"]
    row.append(f"{Am:.1f}")
tab.append(row)

tab.append(comment_row("num. scatters"))
row = []
row.append(r"$M$")
for key in ts:
    num_s = ts[key]["mirror"]["num_scatter_centers"]
    row.append(f"{num_s:d}")
tab.append(row)

tab.append(comment_row("scatter area"))
row = []
row.append(r"$A_\text{mirror-scatter}$\,/\,(cm)$^{2}$")
for key in ts:
    A_mirror_scatter_cm2 = 1e4 * ts[key]["mirror"]["scatter_center_area_m2"]
    row.append(f"{A_mirror_scatter_cm2:.0f}")
tab.append(row)

with open("telescope_table_optics_mirror.tex", "wt") as f:
    f.write(tab_to_latex(tab))

# OPTICS Camera
# =============

tab = []

tab.append(comment_row("diameter"))
row = []
row.append(r"$D_\text{camera}$\,/\,m")
for key in ts:
    Ds = 2 * ts[key]["sensor"]["camera"]["outer_radius_m"]
    row.append(f"{Ds:.2f}")
tab.append(row)

tab.append(comment_row("field-of-view"))
row = []
row.append(r"$\Theta_\text{fov}$\,/\,(1$^\circ$)")
for key in ts:
    Tfov_rad = 2 * np.arctan(
        ts[key]["sensor"]["camera"]["outer_radius_m"]
        / ts[key]["mirror"]["focal_length_m"]
    )
    Tfov_deg = np.rad2deg(Tfov_rad)
    row.append(f"{Tfov_deg:.1f}")
tab.append(row)

tab.append(comment_row("feed horn diameter"))
row = []
row.append(r"$D_\text{feed-horn}$\,/\,cm")
for key in ts:
    Dfhi = 1e2 * 2 * ts[key]["sensor"]["camera"]["feed_horn_inner_radius_m"]
    row.append(f"{Dfhi:.2f}")
tab.append(row)

tab.append(comment_row("feed horn field-of-view"))
row = []
row.append(r"$\Theta_\text{feed-horn-full}$\,/\,(1$^\circ$)")
for key in ts:
    Rfhi_m = ts[key]["sensor"]["camera"]["feed_horn_inner_radius_m"]
    Theta_full_rad = 2 * np.arctan(
        Rfhi_m / ts[key]["mirror"]["focal_length_m"]
    )
    Theta_full_deg = np.rad2deg(Theta_full_rad)
    row.append(f"{Theta_full_deg:.2f}")
tab.append(row)

tab.append(comment_row("num. feed horns"))
row = []
row.append("")
for key in ts:
    num_fh = ts[key]["sensor"]["num_feed_horns"]
    row.append(f"{num_fh:d}")
tab.append(row)

tab.append(comment_row("feed horn area"))
row = []
row.append(r"$A_\text{feed-horn}$\,/\,(cm)$^{2}$")
for key in ts:
    A_fh_cm2 = 1e4 * ts[key]["sensor"]["feed_horn_area_m2"]
    row.append(f"{A_fh_cm2:.2f}")
tab.append(row)

tab.append(comment_row("focus distance"))
row = []
row.append(r"$g$\,/\,km")
for key in ts:
    g_km = 1e-3 * OBJECT_DISTANCE_M
    row.append(f"{g_km:.1f}")
tab.append(row)

tab.append(comment_row("distance to mirror"))
row = []
row.append(r"$d$\,/\,m")
for key in ts:
    d = ts[key]["sensor"]["sensor_distance_m"]
    row.append(f"{d:.2f}")
tab.append(row)

tab.append(comment_row("num. scatters per feed horn"))
row = []
row.append(r"$N$")
for key in ts:
    num_fh = ts[key]["sensor"]["num_scatter_centers_per_feed_horn"]
    row.append(f"{num_fh:d}")
tab.append(row)

tab.append(comment_row("scatter center area"))
row = []
row.append(r"$A_\text{camera-scatter}$\,/\,(cm)$^{2}$")
for key in ts:
    A_fh_scatter_cm2 = (
        1e4 * ts[key]["sensor"]["feed_horn_scatter_center_area_m2"]
    )
    row.append(f"{A_fh_scatter_cm2:.2f}")
tab.append(row)


with open("telescope_table_optics_camera.tex", "wt") as f:
    f.write(tab_to_latex(tab))

# SCATTER
# =======
tab = []
row = []
row.append(r"$M$")
for key in ts:
    num_s = ts[key]["mirror"]["num_scatter_centers"]
    row.append(f"{num_s:d}")
tab.append(row)

row = []
row.append(r"$N$")
for key in ts:
    num_fh = ts[key]["sensor"]["num_feed_horns"]
    row.append(f"{num_fh:d}")
tab.append(row)

row = []
row.append(r"$Ns$")
for key in ts:
    num_sfh = ts[key]["sensor"]["num_scatter_centers_per_feed_horn"]
    row.append(f"{num_sfh:d}")
tab.append(row)

row = []
row.append(r"$\nu_\text{start}$/GHz")
for key in ts:
    nu_start_Hz, _ = iaat.lownoiseblock.input_frequency_start_stop_Hz(
        ts[key]["lnb"]
    )
    row.append(f"{nu_start_Hz *1e-9:.2f}")
tab.append(row)

row = []
row.append(r"$\nu_\text{stop}$/GHz")
for key in ts:
    _, nu_stop_Hz = iaat.lownoiseblock.input_frequency_start_stop_Hz(
        ts[key]["lnb"]
    )
    row.append(f"{nu_stop_Hz *1e-9:.2f}")
tab.append(row)

row = []
row.append(r"$\delta \nu$/MHz")
for key in ts:
    nu_start_Hz, nu_stop_Hz = iaat.lownoiseblock.input_frequency_start_stop_Hz(
        ts[key]["lnb"]
    )
    bandwidth_Hz = nu_stop_Hz - nu_start_Hz
    row.append(f"{bandwidth_Hz *1e-6:.2f}")
tab.append(row)

row = []
row.append(r"$A_\text{antenna}$/(cm)$^{2}$")
for key in ts:
    Aeff_cm2 = 1e4 * ts[key]["lnb"]["effective_area_m2"]
    row.append(f"{Aeff_cm2:.2f}")
tab.append(row)


row = []
row.append(r"$\nu_\text{local-oscillator}$/GHz")
for key in ts:
    nu_LO = ts[key]["lnb"]["local_oscillator_frequency_Hz"]
    nu_LO_std = ts[key]["lnb"]["local_oscillator_frequency_std_Hz"]
    nu_str = iaat.utils.scientific.uncertainty(
        x=nu_LO * 1e-9, dx=nu_LO_std * 1e-9
    )
    row.append("$" + nu_str + "$")
tab.append(row)


row = []
row.append(r"$T_\text{noise}$/K")
for key in ts:
    T = ts[key]["lnb"]["noise_temperature_K"]
    row.append(f"{T:.2f}")
tab.append(row)


row = []
row.append(r"$P_\text{noise}$/pW")
for key in ts:
    P_noise_pW = 1e12 * ts[key]["lnb"]["noise_power_W"]
    row.append(f"{P_noise_pW:.2f}")
tab.append(row)
