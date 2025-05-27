#!/usr/bin/env python3
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import os
import binning_utils


sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])


def ax_add_marker(ax, x, y, s, **kwargs):
    ax.text(
        x=x,
        y=y,
        s=s,
        **kwargs,
        verticalalignment="center",
        horizontalalignment="center",
    )


def airy_full_angle(D, l):
    return 2.0 * iaat.theory.airy_angle(mirror_diameter=D, wavelength=l)


def contour_fmt_deg(x):
    s = f"{x:.2f}"
    return s + r"$^\circ$"


def valid_airy_Nus(theta_D_Nu, D_bin_centers, Nu_bin_centers, theta_threshold):
    Ds = []
    Nus = []
    for iD in range(theta_D_Nu.shape[0]):
        iam = np.argmin(np.abs(theta_D_Nu[iD, :] - theta_threshold))
        Nu = Nu_bin_centers[iam]
        Nus.append(Nu)
        Ds.append(D_bin_centers[iD])
    return np.array(Ds), np.array(Nus)


def ax_add_box_gradient(
    ax, xlim, ylim, alpha_start, alpha_stop, num=255, **kwargs
):
    alphas = np.linspace(alpha_start, alpha_stop, num)
    x_edges = np.linspace(xlim[0], xlim[1], num + 1)
    for i in range(num):
        print(alphas[i])
        ax.fill_between(
            x=[x_edges[i], x_edges[i + 1]],
            y1=[ylim[0], ylim[0]],
            y2=[ylim[1], ylim[1]],
            alpha=alphas[i],
            **kwargs,
        )


def get_mean_input_frequency(telescope):
    lnb = iaat.lownoiseblock.init(key=telescope["lnb_key"])
    nu_start_Hz, nu_stop_Hz = iaat.lownoiseblock.input_frequency_start_stop_Hz(
        lnb=lnb
    )
    return np.mean([nu_start_Hz, nu_stop_Hz])


D_bin = binning_utils.Binning(bin_edges=np.linspace(1, 30, 201))
Nu_bin = binning_utils.Binning(bin_edges=np.geomspace(1e9, 100e9, 301))

D_DEPTH_OF_FIELD_LIMIT_AT_ALTITUDE_2000M = 23.0


lst = iaat.telescopes.init("large_size_telescope")
lst_nu_Hz = get_mean_input_frequency(telescope=lst)
lst_marker = r"$\mathbf{L}$"

mst = iaat.telescopes.init("medium_size_telescope")
mst_nu_Hz = get_mean_input_frequency(telescope=mst)
mst_marker = r"$\mathbf{M}$"

crome = iaat.telescopes.init("crome")
crome_nu_Hz = get_mean_input_frequency(telescope=crome)
crome_marker = r"$\mathbf{C}$"

theta = np.zeros(shape=(D_bin["num"], Nu_bin["num"]))
for iD in range(D_bin["num"]):
    for iNu in range(Nu_bin["num"]):
        theta[iD, iNu] = airy_full_angle(
            D=D_bin["centers"][iD],
            l=iaat.signal.frequency_to_wavelength(
                frequency=Nu_bin["centers"][iNu]
            ),
        )

theta_deg = np.rad2deg(theta)
theta_levels_deg = [
    0.15,
    0.3,
    0.6,
    1.2,
    2.4,
    4.8,
    9.6,
]  # np.geomspace(0.15 / 2, 3.0, 8)

vD, vNu = valid_airy_Nus(
    theta_D_Nu=theta_deg,
    D_bin_centers=D_bin["centers"],
    Nu_bin_centers=Nu_bin["centers"],
    theta_threshold=0.30,
)

# make valid regime polygon
# -------------------------
poly = []
iii = 0
while vD[iii] <= D_DEPTH_OF_FIELD_LIMIT_AT_ALTITUDE_2000M:
    poly.append([vD[iii], vNu[iii]])
    iii += 1
poly.append([vD[iii], Nu_bin["stop"]])
poly.append([min(vD), Nu_bin["stop"]])
poly = np.asarray(poly)

TO_GIGA = 1e-9

fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.8, 0.8])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1, which="both")
contour_label_thingy = ax.contour(
    D_bin["centers"],
    Nu_bin["centers"] * TO_GIGA,
    theta_deg.T,
    levels=theta_levels_deg,
    cmap="Grays_r",
    linewidths=0.5,
)
ax.clabel(
    contour_label_thingy,
    levels=theta_levels_deg,
    fontsize=8,
    fmt=contour_fmt_deg,
)
ax.vlines(
    x=[D_DEPTH_OF_FIELD_LIMIT_AT_ALTITUDE_2000M],
    ymin=Nu_bin["start"] * TO_GIGA,
    ymax=Nu_bin["stop"] * TO_GIGA,
    colors=["black"],
    linestyle="--",
    alpha=0.15,
)
marker_fontsize = 16
ax_add_marker(
    ax=ax,
    x=2.0 * lst["mirror"]["outer_radius_m"],
    y=lst_nu_Hz * TO_GIGA,
    s=lst_marker,
    color="black",
    fontsize=marker_fontsize,
)
ax_add_marker(
    ax=ax,
    x=2.0 * mst["mirror"]["outer_radius_m"],
    y=mst_nu_Hz * TO_GIGA,
    s=mst_marker,
    color="black",
    fontsize=marker_fontsize,
)
ax_add_marker(
    ax=ax,
    x=2.0 * crome["mirror"]["outer_radius_m"],
    y=crome_nu_Hz * TO_GIGA,
    s=crome_marker,
    color="black",
    fontsize=marker_fontsize,
)
fill_alpha = 0.15
HESS2_MIRROR_D_M = 28
ax.fill(poly[:, 0], poly[:, 1] * TO_GIGA, color="black", alpha=fill_alpha)
ax_add_box_gradient(
    ax=ax,
    xlim=[D_DEPTH_OF_FIELD_LIMIT_AT_ALTITUDE_2000M, HESS2_MIRROR_D_M],
    ylim=np.array([min(poly[:, 1]), max(poly[:, 1])]) * TO_GIGA,
    alpha_start=fill_alpha,
    alpha_stop=0.0,
    color="black",
    linewidth=0.0,
)
ax.set_xlim([0.0, D_bin["stop"]])
ax.set_ylim(Nu_bin["limits"] * TO_GIGA)
ax.semilogy()
ax.set_xlabel(r"mirror diameter $D$ / m")
ax.set_ylabel(r"radio frequency $\nu$ / GHz")
fig.savefig("limits_of_mirror_diameter.jpg")
sebplt.close(fig)
