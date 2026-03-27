#!/usr/bin/env python3
import imaging_atmospheric_askaryan_telescope as iaat
import imaging_atmospheric_askaryan_telescope.investigations.airshower_response
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from numpy.fft import rfft, rfftfreq
import os

RANDOM_SEED = 23
work_dir = "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/output_1PeV_50K"
psf_investigation_dir = "/home/anne/Documents/Papers/pet_project/Radio_telescopt_EAS/imaging_atmospheric_askaryan_telescope/imaging_atmospheric_askaryan_telescope/scripts/2025-06-03-psf"

telescope_key = "large_size_telescope"
source_config = iaat.production.radio_from_airshower.make_config()
source_config = {}
source_config["__type__"] = "airshower"
source_config["event_id"] = 1000
source_config["primary_particle"] = {
    "key": "gamma",
    "azimuth_rad": np.deg2rad(0),
    "zenith_rad": np.deg2rad(0),
    "core_north_m": 100.0,
    "core_west_m": 50.0,
    "energy_GeV": 1000000.0,
}
source_config["corsika_coreas_executable_path"] = None

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

ecsf = iaat.calibration.read_energy_conservation_scale_factor(
    path=os.path.join(
        psf_investigation_dir,
        "calibration",
        telescope_key,
        "energy_conservation_scale_factor.json",
    )
)
mirror_to_camera_energy_scale_factor = ecsf["fitted_energy_scale_factor"]

iaat.production.simulate_telescope_response(
    out_dir=work_dir,
    source_config=source_config,
    site=site,
    telescope=telescope,
    timing=timing,
    thermal_noise_random_seed=RANDOM_SEED + 1,
    readout_random_seed=RANDOM_SEED + 2,
    camera_lnb_random_seed=RANDOM_SEED + 3,
    mirror_to_camera_energy_scale_factor=mirror_to_camera_energy_scale_factor,
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

fh = iaat.time_series.read(
    os.path.join(
        work_dir, "feed_horns/electric_fields.tar"
    )  # feed_horns #lnb_signal_output #lnb_signal_and_noise_output
)

positions = telescope["sensor"]["feed_horn_positions_m"]
edge_map = telescope["sensor"]["camera"]["feed_horn_edge_mapping"]
edge_vertices = telescope["sensor"]["camera"]["feed_horn_edge_vertices_m"]


V_to_uV = 1e6
s_to_ns = 1e9

waveforms = fh._x * V_to_uV  # shape: (N_pixels, N_samples, 3)   # µV/m
time_axis = fh.make_time_bin_edges()[:-1] * s_to_ns

xpos = positions[:, 0]
ypos = positions[:, 1]

# --------------------------------------------------
# Camera map quantity (peak field per pixel)
# --------------------------------------------------

sum_pol = waveforms.max(axis=1)

sum_x = sum_pol[:, 0]
sum_y = sum_pol[:, 1]
sum_z = sum_pol[:, 2]
sum_xyz = np.sqrt(sum_x**2 + sum_y**2 + sum_z**2)

# --------------------------------------------------
# Deposited power per pixel (±250 samples around peak)
# --------------------------------------------------

feed_horn_energies_J = iaat.electric_fields.integrate_power_over_time(
    electric_fields=fh,
    channel_effective_area_m2=telescope["mirror"]["area_m2"],
)
feed_horn_energies_eV = feed_horn_energies_J / iaat.signal.ELECTRON_VOLT_J

deposited_power = feed_horn_energies_eV


dt = fh.time_slice_duration_s  # seconds
f_band = (7.0e9, 10.0e9)
E = fh._x  # (N_pix, N_samples, 3)

deposited_power = (
    iaat.investigations.airshower_response.compute_energy_freqband(
        E=E,
        dt=dt,
        f_band=f_band,
        antenna_effective_area_m2=telescope["sensor"]["feed_horn_area_m2"],
    )
)
waveforms = (
    iaat.signal.butter_bandpass_filter(
        amplitudes=E,
        frequency_start=f_band[0],
        frequency_stop=f_band[1],
        time_slice_duration=dt,
    )
    * V_to_uV
)


# --------------------------------------------------
# Build hexagon polygons
# --------------------------------------------------

polygons = [edge_vertices[idx, :2] for idx in edge_map]

plots = [
    (sum_x, "Peak waveform – X pol"),
    (sum_y, "Peak waveform – Y pol"),
    (sum_z, "Peak waveform – Z pol"),
    (deposited_power, "Deposited power"),
]


# --------------------------------------------------
# Camera plot
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

for ax, (data, title) in zip(axes.flat, plots):

    if title == "Deposited power":
        norm = LogNorm(vmin=data.min(), vmax=data.max())
    else:
        norm = None  # linear scale
        # norm = LogNorm(vmin=data.min(), vmax=data.max())

    coll = PolyCollection(
        polygons,
        array=data,
        cmap="Blues",
        edgecolors="k",
        linewidths=0.2,
        norm=norm,
    )

    ax.add_collection(coll)
    ax.set_aspect("equal")
    ax.autoscale_view()

    ax.set_title(title)
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")

    # Add colorbar
    if title == "Deposited power":
        fig.colorbar(
            coll, ax=ax, label=r"Energy / eV"
        )  # r"Deposited power [W/m$^{2}$]")
    else:
        fig.colorbar(coll, ax=ax, label=r"Peak electric field [$\mu$V/m]")

plt.savefig(
    f"{work_dir}/summed_waveforms_per_pixel.png", dpi=300, bbox_inches="tight"
)
plt.close()


fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

# --------------------------------------------------
# Deposited power
# --------------------------------------------------

for data, title in plots:
    if title == "Deposited power":
        deposited_power = data
        break
else:
    raise ValueError("Deposited power not found in plots")

norm = LogNorm(
    vmin=deposited_power[deposited_power > 0].min(), vmax=deposited_power.max()
)

coll = PolyCollection(
    polygons,
    array=deposited_power,
    cmap="Blues",
    edgecolors="k",
    linewidths=0.2,
    norm=norm,
)

ax.add_collection(coll)
ax.set_aspect("equal")
ax.autoscale_view()

# Font sizes
label_fs = 16
tick_fs = 14
cbar_label_fs = 16

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])
ax.set_yticks([])

cbar = fig.colorbar(coll, ax=ax)
cbar.set_label(r"Energy / eV", fontsize=cbar_label_fs)
cbar.ax.tick_params(labelsize=tick_fs)

plt.savefig(f"{work_dir}/deposited_power.png", dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Brightest pixel selection
# --------------------------------------------------

idx_max = np.argmax(sum_xyz)

print("Most illuminated feedhorn index:", idx_max)
print("Position [m]:", positions[idx_max])
print("Peak amplitude [V/m]:", sum_xyz[idx_max])

# --------------------------------------------------
# Time-domain waveform (µV/m)
# --------------------------------------------------

wf = waveforms[idx_max]

wf_mag = np.sqrt(
    wf[:, 0] ** 2
    + wf[:, 1] ** 2  # +
    # wf[:, 2]**2
)

# --------------------------------------------------
# Frequency spectrum
# --------------------------------------------------

dt = fh.time_slice_duration_s  # seconds

# FFT of each band-passed polarization
fft_x = rfft(wf[:, 0])
fft_y = rfft(wf[:, 1])
fft_z = rfft(wf[:, 2])

freqs = rfftfreq(len(wf), dt)

spec_x = np.abs(fft_x)
spec_y = np.abs(fft_y)
spec_z = np.abs(fft_z)

# Optional normalization (for shape comparison only)
spec_x /= spec_x.max()
spec_y /= spec_y.max()
spec_z /= spec_z.max()

Z0 = 376.730313668  # vacuum impedance [Ohm]

n = wf.shape[0]
fs = 1.0 / dt

freqs = rfftfreq(n, dt)

psd = {}

for i_pol, label in enumerate(["x", "y", "z"]):

    # FFT of band-passed signal
    fft_vals = rfft(wf[:, i_pol])

    # One-sided power spectral density
    psd_pol = (np.abs(fft_vals) ** 2) / (Z0 * fs * n)

    psd[label] = psd_pol


# --------------------------------------------------
# Waveform + spectrum plot
# --------------------------------------------------

label_fs = 16
tick_fs = 14
legend_fs = 14

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# ---- Time domain ----
axes[0].plot(time_axis, wf[:, 0], label=r"$E_x$")
axes[0].plot(time_axis, wf[:, 1], label=r"$E_y$")
axes[0].plot(time_axis, wf[:, 2], label=r"$E_z$")
axes[0].set_xlim([110, 130])
axes[0].set_xlabel("Time [ns]", fontsize=label_fs)
axes[0].set_ylabel(r"Electric field [$\mu$V/m]", fontsize=label_fs)
axes[0].legend(fontsize=legend_fs)
axes[0].tick_params(axis="both", which="major", labelsize=tick_fs)

# ---- Frequency domain (individual polarizations) ----
axes[1].plot(freqs, psd["x"], label=r"$E_x$")
axes[1].plot(freqs, psd["y"], label=r"$E_y$")
axes[1].plot(freqs, psd["z"], label=r"$E_z$")
axes[1].set_xlabel("Frequency [Hz]", fontsize=label_fs)
axes[1].set_ylabel(r"PSD [W m$^{-2}$  Hz$^{-1}$]", fontsize=label_fs)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].legend(fontsize=legend_fs)
axes[1].tick_params(axis="both", which="major", labelsize=tick_fs)

plt.savefig(
    f"{work_dir}/brightest_pixel_waveform_and_spectrum.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
