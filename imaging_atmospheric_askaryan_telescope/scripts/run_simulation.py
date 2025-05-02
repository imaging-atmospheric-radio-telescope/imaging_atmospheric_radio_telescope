#!/usr/bin/env python3
import argparse

import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import numpy as np
import json_utils
import os

work_dir = "run"

if not os.path.exists(work_dir):
    iaat.init(
        work_dir=work_dir,
        site_key="namibia",
        telescope_key="large_size_telescope",
    )

askaryan = iaat.from_config(work_dir=work_dir)
telescope = askaryan["telescope"]
timing = askaryan["timing"]
site = askaryan["site"]


random_seed = 1405

if False:
    source_config = iaat.production.radio_from_airshower.make_config()
    source_config["event_id"] = random_seed
    source_config["primary_particle"]["key"] = "gamma"
    source_config["primary_particle"]["azimuth_rad"] = np.deg2rad(30)
    source_config["primary_particle"]["zenith_rad"] = np.deg2rad(1.5)
    source_config["primary_particle"]["core_north_m"] = 50.0
    source_config["primary_particle"]["core_west_m"] = 20.0
    source_config["primary_particle"]["energy_GeV"] = 10_000.0
else:
    source_config = iaat.production.radio_from_plane_wave.make_config()
    source_config["geometry"]["azimuth_rad"] = np.deg2rad(220)
    source_config["geometry"]["zenith_rad"] = np.deg2rad(1.7)
    source_config["geometry"][
        "distance_to_plane_defining_time_zero_m"
    ] = iaat.corsika.TOP_OF_ATMOSPHERE_ALTITUDE_M
    source_config["power"]["power_of_isotrop_and_point_like_emitter_W"] = 2e-1
    source_config["power"][
        "distance_to_isotrop_and_point_like_emitter_m"
    ] = 100e3
    source_config["sine_wave"]["emission_frequency_Hz"] = 11.1e9
    source_config["sine_wave"]["emission_duration_s"] = 5e-9
    source_config["sine_wave"]["emission_ramp_up_duration_s"] = 1e-9
    source_config["sine_wave"]["emission_ramp_down_duration_s"] = 1e-9

out_dir = os.path.join(
    work_dir, source_config["__type__"], f"{random_seed:06d}"
)

# start simulation
# ----------------
prng = np.random.Generator(np.random.PCG64(random_seed))

iaat.production.simulate_telescope_response(
    out_dir=out_dir,
    source_config=source_config,
    site=site,
    telescope=telescope,
    timing=timing,
    thermal_noise_random_seed=random_seed + 1,
)

plot_dir = os.path.join(out_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)

# plot instrument
# ---------------
pos_keys = {
    "mirror": "scatter_center_positions_m",
    "sensor": "feed_horn_positions_m",
}
for component in ["mirror", "sensor"]:
    fig_path = os.path.join(plot_dir, component + ".antenna_positions.jpg")
    if not os.path.exists(fig_path):
        print("plot", fig_path)
        iaat_plot.write_figure_antenna_positions(
            positions=telescope[component][pos_keys[component]], path=fig_path
        )


# plot electic fields
# -------------------
for component in ["probe", "mirror", "feed_horns"]:
    if component == "feed_horns":
        channels_label = "feed horns / 1"
        A_effective_m2 = (
            1 / telescope["sensor"]["feed_horn_areal_density_per_m2"]
        )
        roi_frequency = [2.5e9, 25e9]
    elif component == "mirror":
        channels_label = "scatter centers / 1"
        roi_frequency = [2.5e9, 25e9]
        A_effective_m2 = (
            1 / telescope["mirror"]["scatter_center_areal_density_per_m2"]
        )
    else:
        channels_label = "channels / 1"
        roi_time = None
        roi_frequency = None
        A_effective_m2 = iaat.signal.calculate_antenna_effective_area(
            wavelength=iaat.signal.frequency_to_wavelength(
                frequency=telescope["lnb"]["local_oscillator_frequency_Hz"]
            ),
            gain=1.0,
        )

    field_path = os.path.join(out_dir, component, "electric_fields.tar")

    # Electric fields
    # ---------------
    fig_path = os.path.join(plot_dir, f"{component:s}_electric_fields.jpg")
    if not os.path.exists(fig_path) and os.path.exists(field_path):
        print("plot", fig_path)
        field = iaat.electric_fields.read_tar(field_path)
        iaat_plot.write_figure_electric_fields_overview(
            electric_fields=field,
            path=fig_path,
            component_mask=[1, 1, 0],
            channels_label=channels_label,
            figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
            norm=None,
            vmin=np.max(field["electric_fields_V_per_m"]),
            vmax=1e6 * np.max(field["electric_fields_V_per_m"]),
        )

    # Areal power density
    # -------------------
    fig_path = os.path.join(out_dir, "plot", f"{component}_power_density.jpg")
    if not os.path.exists(fig_path) and os.path.exists(field_path):
        print("plot", fig_path)
        field = iaat.electric_fields.read_tar(field_path)

        field_norm = np.linalg.norm(field["electric_fields_V_per_m"], axis=2)
        assert field_norm.shape[0] == field["num_antennas"]
        assert field_norm.shape[1] == field["num_time_slices"]

        _power = iaat.signal.calculate_antenna_power(
            effective_area=A_effective_m2,
            electric_field=field_norm,
        )
        _power_density = _power / A_effective_m2

        iaat_plot.write_matrix(
            path=fig_path,
            matrix=_power_density,
            x_bin_edges=1e9 * iaat.electric_fields.make_time_bin_edges(field),
            y_bin_edges=iaat.electric_fields.make_antenna_bin_edges(field),
            x_label="time / ns",
            y_label=channels_label,
            z_label=r"areal power density / Wm$^{-2}$",
            cmap="viridis",
            cmap_marker=None,
            norm=iaat_plot.seb.matplotlib.colors.LogNorm(),
            figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
            title=None,
        )

    frequency_bin_edges_Hz = np.geomspace(2.5e9, 25e9, 31)

    fig_path = os.path.join(
        out_dir, "plot", f"{component}_power_density_spectrum.jpg"
    )
    if not os.path.exists(fig_path) and os.path.exists(field_path):
        print("plot", fig_path)
        field = iaat.electric_fields.read_tar(field_path)

        Prho_W_per_Hz_per_m2 = iaat.electric_fields.estimate_power_spectrum_density_W_per_Hz_per_m2(
            electric_fields=field,
            antenna_effective_area_m2=A_effective_m2,
            frequency_bin_edges_Hz=frequency_bin_edges_Hz,
        )
        iaat_plot.write_matrix(
            path=fig_path,
            matrix=Prho_W_per_Hz_per_m2.T,
            x_bin_edges=1e-9 * frequency_bin_edges_Hz,
            y_bin_edges=iaat.electric_fields.make_antenna_bin_edges(field),
            x_label=r"frequency $\nu$ / GHz",
            y_label=channels_label,
            z_label=r"power density / W m$^{-2}$ (Hz)$^{-1}$",
            cmap="viridis",
            cmap_marker=None,
            norm=iaat_plot.seb.matplotlib.colors.LogNorm(),
            figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
            title=None,
        )


# check energy conservation
# -------------------------
if True:
    E_mirror_path = os.path.join(out_dir, "mirror", "electric_fields.tar")
    E_mirror = iaat.electric_fields.read_tar(E_mirror_path)

    E_feed_horns_path = os.path.join(
        out_dir, "feed_horns", "electric_fields.tar"
    )
    E_feed_horns = iaat.electric_fields.read_tar(E_feed_horns_path)

    A_eff_mirror_scatter_center_m2 = (
        1.0 / telescope["mirror"]["scatter_center_areal_density_per_m2"]
    )
    A_eff_sensor_feed_horn_m2 = (
        1.0 / telescope["sensor"]["feed_horn_areal_density_per_m2"]
    )

    P_mirror_W = np.zeros(
        shape=(E_mirror["num_antennas"], E_mirror["num_time_slices"])
    )
    P_sensor_W = np.zeros(
        shape=(E_feed_horns["num_antennas"], E_feed_horns["num_time_slices"])
    )

    for dim in [0, 1]:
        P_mirror_W_dim = iaat.signal.calculate_antenna_power(
            effective_area=A_eff_mirror_scatter_center_m2,
            electric_field=E_mirror["electric_fields_V_per_m"][:, :, dim],
        )
        P_mirror_W += P_mirror_W_dim

        P_sensor_W_dim = iaat.signal.calculate_antenna_power(
            effective_area=A_eff_sensor_feed_horn_m2,
            electric_field=E_feed_horns["electric_fields_V_per_m"][:, :, dim],
        )
        P_sensor_W += P_sensor_W_dim

    En_mirror_J = np.sum(P_mirror_W) * E_mirror["time_slice_duration_s"]
    En_sensor_J = np.sum(P_sensor_W) * E_feed_horns["time_slice_duration_s"]

    En_mirror_eV = En_mirror_J / iaat.signal.ELECTRON_VOLT_J
    En_sensor_eV = En_sensor_J / iaat.signal.ELECTRON_VOLT_J

    if source_config["__type__"] == "plane_wave":
        _power_geom = (
            iaat.calibration_source.plane_wave_in_far_field.make_power_setup(
                **source_config["power"]
            )
        )
        expected_power_on_mirror_W = (
            telescope["mirror"]["area_m2"]
            * _power_geom["pointing_vector_magnitude_W_per_m2"]
        )
        expected_energy_on_mirror_J = (
            expected_power_on_mirror_W
            * source_config["sine_wave"]["emission_duration_s"]
        )
        expected_energy_on_mirror_eV = (
            expected_energy_on_mirror_J / iaat.signal.ELECTRON_VOLT_J
        )

        print(
            "Energy on mirror",
            En_mirror_eV,
            "eV",
            "expected",
            expected_energy_on_mirror_eV,
            "eV",
        )
        print("Energy on feed horns", En_sensor_eV, "eV")


# plot lnb mixer gain
# -------------------
fig_path = os.path.join(plot_dir, "lnb_input_gain.jpg")
if not os.path.exists(fig_path):
    _lnb_bench_frequency_Hz = np.geomspace(8e9, 16e9, 100)
    _lnb_bench_gain = iaat.signal.butter_bench(
        frequencies=_lnb_bench_frequency_Hz,
        bandpass=iaat.signal.butter_bandpass_filter,
        filter_config={
            "frequency_start": telescope["lnb"][
                "local_oscillator_frequency_Hz"
            ]
            + telescope["lnb"]["intermediate_frequency_start_Hz"],
            "frequency_stop": telescope["lnb"]["local_oscillator_frequency_Hz"]
            + telescope["lnb"]["intermediate_frequency_stop_Hz"],
        },
        num_time_slices=10000,
        time_slice_duration=timing["electric_fields"]["time_slice_duration_s"],
    )
    iaat_plot.write_figure_gain(
        path=fig_path,
        frequency=_lnb_bench_frequency_Hz,
        gain=_lnb_bench_gain,
        scale="G",
        frequency_lim=[8e9, 16e9],
    )

fig_path = os.path.join(plot_dir, "lnb_output_gain.jpg")
if not os.path.exists(fig_path):
    _lnb_bench_frequency_Hz = np.geomspace(0.1e9, 10e9, 100)
    _lnb_bench_gain = iaat.signal.butter_bench(
        frequencies=_lnb_bench_frequency_Hz,
        bandpass=iaat.signal.butter_bandpass_filter,
        filter_config={
            "frequency_start": telescope["lnb"][
                "intermediate_frequency_start_Hz"
            ],
            "frequency_stop": telescope["lnb"][
                "intermediate_frequency_stop_Hz"
            ],
        },
        num_time_slices=10000,
        time_slice_duration=timing["electric_fields"]["time_slice_duration_s"],
    )
    iaat_plot.write_figure_gain(
        path=fig_path,
        frequency=_lnb_bench_frequency_Hz,
        gain=_lnb_bench_gain,
        scale="G",
        frequency_lim=[1e8, 1e10],
    )


E_feed_horns = iaat.electric_fields.read_tar(
    os.path.join(out_dir, "feed_horns", "electric_fields.tar")
)

E_lnb_output = iaat.electric_fields.read_tar(
    os.path.join(out_dir, "lnb_signal_and_noise_output", "electric_fields.tar")
)

# efield to power
total_power_leaving_lnb = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area_m2"],
    electric_field=E_lnb_output["electric_fields_V_per_m"],
)
numS = E_feed_horns["num_time_slices"]


# plot power_leaving_lnb
# ----------------------
fig_path_power_leaving_lnb = os.path.join(plot_dir, "lnb_output.jpg")
if not os.path.exists(fig_path_power_leaving_lnb):
    total_power_leaving_lnb_xy = np.sum(
        total_power_leaving_lnb[:, numS:, 0:2],
        axis=2,
    )
    pixel_bin_edges = iaat.electric_fields.make_antenna_bin_edges(
        electric_fields=E_feed_horns,
    )
    time_bin_edges = iaat.electric_fields.make_time_bin_edges(
        electric_fields=E_feed_horns,
        global_time=False,
    )
    iaat_plot.write_figure_lnb_power(
        path=fig_path_power_leaving_lnb,
        lnb_power_W=total_power_leaving_lnb_xy,
        channels_bin_edges=pixel_bin_edges,
        relative_time_bin_edges_s=time_bin_edges,
        global_start_time_s=E_feed_horns["global_start_time_s"],
        lnb_power_min_fraction_of_max=1e-3,
        norm=None,
        lnb_power_min_W=1e-6 * np.max(total_power_leaving_lnb_xy),
        lnb_power_max_W=np.max(total_power_leaving_lnb_xy),
        expected_noise_power_W=telescope["lnb"]["noise_power_W"],
        channels_label="pixels / 1",
        figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
        roi_time=[5e-9, 10e-9],
    )

# integrate power_leaving_lnb over time for readout
# -------------------------------------------------
total_power_sliding_integral = np.zeros(shape=total_power_leaving_lnb.shape)

numT = timing["readout"]["integrates_num_simulation_time_slices"]
simulation_time_slice_duration = timing["electric_fields"][
    "time_slice_duration_s"
]

for t in range(E_lnb_output["num_time_slices"] - numT):
    w = np.sum(total_power_leaving_lnb[:, t : t + numT, :], axis=1)
    total_power_sliding_integral[:, t, :] = w * simulation_time_slice_duration

simulation_time_slices_which_are_sampled_by_readout = np.arange(
    0,
    E_lnb_output["num_time_slices"],
    numT,
)
random_offset_or_readout_wrt_global_time_num_time_slices = int(
    prng.uniform(low=0, high=numT)
)
num_readout_frames = (
    len(simulation_time_slices_which_are_sampled_by_readout) - 1
)
readout_energy = np.zeros(
    shape=(telescope["sensor"]["num_feed_horns"], num_readout_frames, 2)
)
readout_global_start_time_s = (
    random_offset_or_readout_wrt_global_time_num_time_slices
    * timing["electric_fields"]["time_slice_duration_s"]
    + E_lnb_output["global_start_time_s"]
)
for i in range(num_readout_frames):
    simulation_time_slice = (
        simulation_time_slices_which_are_sampled_by_readout[i]
    )
    simulation_time_slice += (
        random_offset_or_readout_wrt_global_time_num_time_slices
    )
    x_comp_energy = total_power_sliding_integral[:, simulation_time_slice, 0]
    y_comp_energy = total_power_sliding_integral[:, simulation_time_slice, 1]
    readout_energy[:, i, 0] = x_comp_energy
    readout_energy[:, i, 1] = y_comp_energy

# plot readout gain
# -----------------
_readout_bench_f_stop = 0.9 * (
    timing["electric_fields"]["sampling_frequency_Hz"]
    / timing["readout"]["integrates_num_simulation_time_slices"]
)
_readout_bench_f_start = 1e-2 * _readout_bench_f_stop
_readout_bench_frequency = np.geomspace(
    _readout_bench_f_start, _readout_bench_f_stop, 100
)
_readout_bench_gain = np.zeros(_readout_bench_frequency.shape)
for _i, _ff in enumerate(_readout_bench_frequency):
    _t, _Ain = iaat.signal.make_sin(
        frequency=_ff,
        time_slice_duration=timing["electric_fields"]["time_slice_duration_s"],
        num_time_slices=1000 * 10,
    )
    _Aout = iaat.signal.integrate_sliding_window(
        signal=_Ain,
        time_slice_duration=1 / numT,
        window_num_slices=numT,
    )
    _r = np.sum(_Aout**2) / np.sum(_Ain**2)
    _readout_bench_gain[_i] = _r
_fig_path_readout_gain = os.path.join(plot_dir, "readout_gain.jpg")
if not os.path.exists(_fig_path_readout_gain):
    iaat_plot.write_figure_gain(
        path=_fig_path_readout_gain,
        frequency=_readout_bench_frequency,
        gain=_readout_bench_gain,
        scale="M",
    )

# plot images seen by readout
# ---------------------------
for units in ["electron_volt", "black_body_temperature", "jansky"]:
    plot_sensor_dir = os.path.join(plot_dir, "readout", units)
    if not os.path.exists(plot_sensor_dir):
        iaat_plot.save_image_slices_energy_deposite(
            readout_energy_J=readout_energy,
            readout_time_slice_duration_s=timing["readout"][
                "time_slice_duration_s"
            ],
            antenna_positions=np.rad2deg(
                telescope["sensor"]["feed_horn_positions_m"]
                / telescope["mirror"]["focal_length_m"]
            ),
            path=plot_sensor_dir,
            global_start_time_s=readout_global_start_time_s,
            units=units,
            bandwidth_Hz=telescope["lnb"]["intermediate_bandwidth_Hz"],
            mirror_area_m2=telescope["mirror"]["area_m2"],
            image_x_label="$c_x$ / (1$^{\circ}$)",
            image_y_label="$c_y$ / (1$^{\circ}$)",
        )
