#!/usr/bin/env python3

import os
import json
from collections import defaultdict

import imaging_atmospheric_askaryan_telescope as iaat
import imaging_atmospheric_askaryan_telescope.investigations.airshower_response

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable

# ==================================================
# User configuration
# ==================================================

work_dirs = [
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_5",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_6",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_7",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_8",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_9",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_10",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_11",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_12",
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_13",
]

psf_investigation_dir = (
    "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/"
    "imaging_atmospheric_askaryan_telescope/"
    "imaging_atmospheric_askaryan_telescope/scripts/2025-06-03-psf"
)

telescope_key = "large_size_telescope"

# Font sizes
label_fs = 18
tick_fs = 14
cbar_label_fs = 16

# Fixed particle order (columns)
particle_order = ["gamma", "proton", "iron"]

particle_cmaps = {
    "gamma": ("Blues", r"$\gamma$"),
    "proton": ("Blues", r"$p$"),
    "iron": ("Blues", r"Fe"),
}


# ==================================================
# Telescope & geometry (EXACTLY like your working script)
# ==================================================

site = iaat.sites.init("karlsruhe")

config = iaat.investigations.point_spread_function.utils.read_config(
    psf_investigation_dir
)

telescope, timing, _ = (
    iaat.investigations.point_spread_function.utils.make_telescope_timing_and_site(
        work_dir=psf_investigation_dir,
        config=config,
        telescope_key=telescope_key,
        sensor_distance_m=None,
    )
)

edge_map = telescope["sensor"]["camera"]["feed_horn_edge_mapping"]
edge_vertices = telescope["sensor"]["camera"]["feed_horn_edge_vertices_m"]

polygons = [edge_vertices[idx, :2] for idx in edge_map]


# ==================================================
# Load simulations and group by energy / particle
# ==================================================

# data[energy_TeV][particle] = E_amp_array
data = defaultdict(dict)
energies = set()

for work_dir in work_dirs:

    with open(os.path.join(work_dir, "source_config.json"), "r") as f:
        source_config = json.load(f)

    particle = source_config["primary_particle"]["key"]

    energy_GeV = source_config["primary_particle"]["energy_GeV"]
    energy_TeV = energy_GeV / 1e3

    fh = iaat.time_series.read(
        os.path.join(work_dir, "feed_horns/electric_fields.tar")
    )

    # E_amp = compute_energy(fh)

    dt = fh.time_slice_duration_s  # seconds

    E = fh._x  # (N_pix, N_samples, 3)

    E_amp = iaat.investigations.airshower_response.compute_energy_freqband(
        E=E,
        dt=dt,
        f_band=(3.0e9, 10.0e9),
        antenna_effective_area_m2=telescope["sensor"]["feed_horn_area_m2"],
    )

    data[energy_TeV][particle] = E_amp

    energies.add(energy_TeV)

energy_list = sorted(energies)


# ==================================================
# Plot grid: rows = energy, columns = particle
# One colorbar per row
# ==================================================

n_rows = len(energy_list)
n_cols = len(particle_order)

fig = plt.figure(figsize=(4.8 * n_cols, 3.5 * n_rows))
gs = gridspec.GridSpec(
    n_rows,
    n_cols,
    wspace=-0.2,
    hspace=0.1,
)


for i, energy in enumerate(energy_list):

    row_E = np.concatenate(
        [data[energy][p] for p in particle_order if p in data[energy]]
    )

    for j, particle in enumerate(particle_order):

        ax = fig.add_subplot(gs[i, j])

        if particle not in data[energy]:
            ax.axis("off")
            continue

        E_amp = data[energy][particle]
        cmap, label = particle_cmaps[particle]

        norm = LogNorm(vmin=E_amp[E_amp > 0].min(), vmax=E_amp.max())
        coll = PolyCollection(
            polygons,
            array=E_amp,
            cmap=cmap,
            norm=norm,
            edgecolors="k",
            linewidths=0.2,
        )

        ax.add_collection(coll)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.tick_params(labelsize=tick_fs)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(coll, ax=ax, fraction=0.038, pad=0.01)
        cbar.ax.tick_params(labelsize=tick_fs)

        if i == 0:
            ax.set_title(label, fontsize=label_fs)

        if j == 0:
            ax.set_ylabel(f"{energy:.0f} TeV", fontsize=label_fs)

        if j == n_cols - 1:
            cbar.set_label(r"Energy / eV", fontsize=cbar_label_fs)


# ==================================================
# Save
# ==================================================

plt.savefig(
    "electric_field_amplitude_energy_particle_grid.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
