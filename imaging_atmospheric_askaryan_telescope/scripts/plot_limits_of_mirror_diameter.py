#!/usr/bin/env python3
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import os
import binning_utils


sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])


def airy_theta(D, l):
    return np.arcsin(1.22 * l / D)


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


D_bin = binning_utils.Binning(bin_edges=np.linspace(3, 30, 201))
Nu_bin = binning_utils.Binning(bin_edges=np.geomspace(1e9, 100e9, 301))

D_DEPTH_OF_FIELD_LIMIT_AT_ALTITUDE_2000M = 23.0

NU_ASTRA_UNIVERSAL_HZ = 9.75e9

theta = np.zeros(shape=(D_bin["num"], Nu_bin["num"]))
for iD in range(D_bin["num"]):
    for iNu in range(Nu_bin["num"]):
        theta[iD, iNu] = airy_theta(
            D=D_bin["centers"][iD],
            l=iaat.signal.frequency_to_wavelength(
                frequency=Nu_bin["centers"][iNu]
            ),
        )

theta_deg = np.rad2deg(theta)
theta_levels_deg = [
    0.067,
    0.15,
    0.3,
    0.6,
    1.2,
    2.4,
]  # np.geomspace(0.15 / 2, 3.0, 8)

vD, vNu = valid_airy_Nus(
    theta_D_Nu=theta_deg,
    D_bin_centers=D_bin["centers"],
    Nu_bin_centers=Nu_bin["centers"],
    theta_threshold=0.15,
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

fig = sebplt.figure(style={"rows": 720, "cols": 1280, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.2, 0.8, 0.75])
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
    alpha=0.33,
)
# ax.plot(vD, vNu, "r")
ax.plot(
    D_DEPTH_OF_FIELD_LIMIT_AT_ALTITUDE_2000M,
    NU_ASTRA_UNIVERSAL_HZ * TO_GIGA,
    marker="o",
    color="black",
)
ax.fill(poly[:, 0], poly[:, 1] * TO_GIGA, color="black", alpha=0.15)
ax.set_xlim(D_bin["limits"])
ax.set_ylim(Nu_bin["limits"] * TO_GIGA)
ax.semilogy()
ax.set_xlabel(r"mirror diameter $D$ / m")
ax.set_ylabel(r"radio frequency $\nu$ / GHz")
fig.savefig("limits_of_mirror_diameter.jpg")
sebplt.close(fig)
