#!/usr/bin/env python3
import argparse

import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import numpy as np
import json_numpy
import os


def read_dict(path):
    with open(path, "rt") as f:
        config = json_numpy.loads(f.read())
    return config


def write_and_read_back_dict(path, config):
    with open(path, "wt") as f:
        f.write(json_numpy.dumps(config, indent=4))
    return read_dict(path)


parser = argparse.ArgumentParser(description="Simulate Askaryan-telescope")

parser.add_argument(
    "-o", help="Path to output directory", required=True, metavar="OUT_DIR"
)
parser.add_argument(
    "-n", type=int, help="unique identifier", required=True, metavar="ID"
)
parser.add_argument(
    "-p",
    help="Path to particle config-file (json).",
    required=True,
    metavar="PARTICLE_PATH",
)
parser.add_argument(
    "-i",
    help="Path to instrument config-file (json).",
    required=True,
    metavar="INSTRUMENT_PATH",
)
parser.add_argument(
    "-c",
    help="Path to CORSIKA executable",
    required=False,
    metavar="CORSIKA_PATH",
    default=os.path.join(
        "build",
        "corsika-77100",
        "run",
        "corsika77100Linux_QGSII_urqmd_coreas",
    ),
)

args = parser.parse_args()
random_seed = args.n
corsika_coreas_executable_path = args.c
config = read_dict(path=args.i)
primary_particle = read_dict(path=args.p)

out_dir = os.path.join(args.o, "{:06d}".format(random_seed))

if os.path.exists(out_dir):
    config = read_dict(path=os.path.join(out_dir, "config.json"),)
    primary_particle = read_dict(path=os.path.join(out_dir, "primary.json"),)
    random_seed = read_dict(path=os.path.join(out_dir, "random_seed.json"))["random_seed"]
else:
    os.makedirs(out_dir, exist_ok=True)
    config = write_and_read_back_dict(
        path=os.path.join(out_dir, "config.json"), config=config,
    )
    primary_particle = write_and_read_back_dict(
        path=os.path.join(out_dir, "primary.json"), config=primary_particle,
    )
    random_seed = write_and_read_back_dict(
        path=os.path.join(out_dir, "random_seed.json"), config={"random_seed": random_seed},
    )["random_seed"]

# init
# ----

telescope, timing = iaat.init_telescope_and_timing(config=config)
site = iaat.sites.init(site_name=config["site_name"])


# start simulation
# ----------------

prng = np.random.Generator(np.random.PCG64(random_seed))


iaat.production.simulate_telescope_response(
    corsika_coreas_executable_path=corsika_coreas_executable_path,
    out_dir=out_dir,
    event_id=random_seed,
    primary_particle=primary_particle,
    site=site,
    telescope=telescope,
    timing=timing,
)

# plot electic fields
# -------------------
plot_dir = os.path.join(out_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)
for component in ["probe", "mirror", "sensor"]:
    if component == "sensor":
        channels_label = "pixels / 1"
        roi_time = [2.5e-9, 7.5e-9]
        roi_frequency = [2.5e9, 25e9]
    elif component == "mirror":
        channels_label = "scatter-centers / 1"
        roi_time = [2.5e-9, 7.5e-9]
        roi_frequency = [2.5e9, 25e9]
    else:
        channels_label = "channels / 1"
        roi_time = None
        roi_frequency = None

    fig_path = os.path.join(plot_dir, component + ".jpg")
    if not os.path.exists(fig_path):
        field_path = os.path.join(out_dir, component, "electric_fields.tar")
        field = iaat.electric_fields.read_tar(field_path)
        iaat_plot.write_figure_electric_fields_overview(
            electric_fields=field,
            path=fig_path,
            component_mask=[1, 1, 0],
            channels_label=channels_label,
            figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
            norm=None,
            vmin=0.0,
            vmax=np.max(field["electric_fields_V_per_m"]),
            roi_time=roi_time,
        )

    fig_spectrum_path = os.path.join(
        plot_dir, component + "_power_spectrum_density.jpg"
    )
    if not os.path.exists(fig_spectrum_path):
        field_path = os.path.join(out_dir, component, "electric_fields.tar")
        field = iaat.electric_fields.read_tar(field_path)

        iaat_plot.write_figure_electric_fields_power_density_spectrum(
            path=fig_spectrum_path,
            electric_fields=field,
            component_mask=[1, 1, 0],
            num_time_slices_to_average_over=(
                field["electric_fields_V_per_m"].shape[1] // 20
            ),
            channels_label=channels_label,
            figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
            roi_frequency=roi_frequency,
        )


# check energy conservation
# -------------------------
COOLING = 1e-2
if True:
    E_mirror_path = os.path.join(out_dir, "mirror", "electric_fields.tar")
    E_mirror = iaat.electric_fields.read_tar(E_mirror_path)

    E_sensor_path = os.path.join(out_dir, "sensor", "electric_fields.tar")
    E_sensor = iaat.electric_fields.read_tar(E_sensor_path)

    A_eff_mirror_scatter_center_m2 = 1.0 / telescope["mirror"]["scatter_center_areal_density_per_m2"]
    A_eff_sensor_feed_horn_m2 = 1.0 / telescope["sensor"]["feed_horn_areal_density_per_m2"]

    scatter_gain = A_eff_mirror_scatter_center_m2 / A_eff_sensor_feed_horn_m2

    P_mirror_W = np.zeros(shape=(E_mirror["num_antennas"], E_mirror["num_time_slices"]))
    P_sensor_W = np.zeros(shape=(E_sensor["num_antennas"], E_sensor["num_time_slices"]))

    for dim in [0, 1]:
        P_mirror_W_dim = iaat.signal.calculate_antenna_power(
            effective_area=A_eff_mirror_scatter_center_m2,
            electric_field=E_mirror["electric_fields_V_per_m"][:, :, dim]
        )
        P_mirror_W += P_mirror_W_dim

        P_sensor_W_dim = iaat.signal.calculate_antenna_power(
            effective_area=A_eff_sensor_feed_horn_m2,
            electric_field=E_sensor["electric_fields_V_per_m"][:, :, dim]
        )
        P_sensor_W += P_sensor_W_dim

    En_mirror_J = np.sum(P_mirror_W) * E_mirror["time_slice_duration_s"]
    En_sensor_J = np.sum(P_sensor_W) * E_sensor["time_slice_duration_s"]

    print("Energy on mirror", 1e6 * En_mirror_J/iaat_plot.ELECTRON_VOLT_J, "ueV")
    print("Energy on sensor", 1e6 * En_sensor_J/iaat_plot.ELECTRON_VOLT_J, "ueV")


# plot instrument
# ---------------
pos_keys = {
    "mirror": "scatter_center_positions_m",
    "sensor": "feed_horn_positions_m",
}
for component in ["mirror", "sensor"]:
    fig_path = os.path.join(plot_dir, component + ".antenna_positions.jpg")
    if not os.path.exists(fig_path):
        iaat_plot.write_figure_antenna_positions(
            positions=telescope[component][pos_keys[component]], path=fig_path
        )

# simulate lnb
# ------------
feed_horn_geometric_gain = (
    telescope["sensor"]["feed_horn_area_m2"]
    / telescope["lnb"]["effective_area_m2"]
)
feed_horn_gain = (
    feed_horn_geometric_gain
    * telescope["sensor"]["feed_horn_transmission"]
)
sensor_electric_fields = iaat.electric_fields.read_tar(
    path=os.path.join(out_dir, "sensor", "electric_fields.tar")
)

signal_efield_entering_lnb = (
    np.sqrt(feed_horn_gain) * sensor_electric_fields["electric_fields_V_per_m"]
)
signal_efield_leaving_lnb = iaat.signal.lnb_mixer(
    amplitudes=signal_efield_entering_lnb,
    time_slice_duration=timing["electric_fields"]["time_slice_duration_s"],
    local_oscillator_frequency=telescope["lnb"][
        "local_oscillator_frequency_Hz"
    ],
    intermediate_frequency_start=telescope["lnb"][
        "intermediate_frequency_start_Hz"
    ],
    intermediate_frequency_stop=telescope["lnb"][
        "intermediate_frequency_stop_Hz"
    ],
)

# plot lnb mixer gain
_lnb_bench_frequency_Hz = np.geomspace(0.1e9, 10e9, 100)
_lnb_bench_gain = iaat.signal.butter_bench(
    frequencies=_lnb_bench_frequency_Hz,
    bandpass=iaat.signal.butter_bandpass_filter,
    filter_config={
        "frequency_start": telescope["lnb"]["intermediate_frequency_start_Hz"],
        "frequency_stop": telescope["lnb"]["intermediate_frequency_stop_Hz"],
    },
    num_time_slices=10000,
    time_slice_duration=timing["electric_fields"]["time_slice_duration_s"],
)
_fig_path_lnb_gain = os.path.join(plot_dir, "lnb_gain.jpg")
if not os.path.exists(_fig_path_lnb_gain):
    iaat_plot.write_figure_gain(
        path=_fig_path_lnb_gain,
        frequency=_lnb_bench_frequency_Hz,
        gain=_lnb_bench_gain,
    )

# thermal noise
# -------------
electric_field_thermal_noise_amplitude_V_per_m = iaat.signal.electric_field_of_thermal_noise(
    antenna_temperature_K=telescope["lnb"]["noise_temperature_K"]*COOLING,
    antenna_bandwidth=telescope["lnb"]["intermediate_bandwidth_Hz"],
)

noise_num_time_slices = int(sensor_electric_fields["num_time_slices"]) * int(2)
lnb_simulation_global_start_time_s = (
    sensor_electric_fields["global_start_time_s"]
    - sensor_electric_fields["num_time_slices"]
    * sensor_electric_fields["time_slice_duration_s"]
)

noise_efield_leaving_lnb = np.sqrt(
    1 / telescope["lnb"]["effective_area_m2"]
) * prng.normal(
    loc=0.0,
    scale=electric_field_thermal_noise_amplitude_V_per_m,
    size=(sensor_electric_fields["num_antennas"], noise_num_time_slices, 3,),
)

_noise_power_W = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area_m2"],
    electric_field=noise_efield_leaving_lnb,
)

assert (
    0.9 < (telescope["lnb"]["noise_power_W"]*COOLING / np.mean(_noise_power_W)) < 1.1
)

# adding signal and noise
numS = sensor_electric_fields["num_time_slices"]
total_efield_leaving_lnb = noise_efield_leaving_lnb
total_efield_leaving_lnb[:, numS:, :] += signal_efield_leaving_lnb

# efield to power
total_power_leaving_lnb = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area_m2"],
    electric_field=total_efield_leaving_lnb,
)

# plot power_leaving_lnb
# ----------------------
fig_path_power_leaving_lnb = os.path.join(plot_dir, "lnb_output.jpg")
if not os.path.exists(fig_path_power_leaving_lnb):
    total_power_leaving_lnb_xy = np.sum(
        total_power_leaving_lnb[:, numS:, 0:2], axis=2,
    )
    pixel_bin_edges = iaat.electric_fields.make_antenna_bin_edges(
        electric_fields=sensor_electric_fields,
    )
    time_bin_edges = iaat.electric_fields.make_time_bin_edges(
        electric_fields=sensor_electric_fields, global_time=False,
    )
    iaat_plot.write_figure_lnb_power(
        path=fig_path_power_leaving_lnb,
        lnb_power_W=total_power_leaving_lnb_xy,
        channels_bin_edges=pixel_bin_edges,
        relative_time_bin_edges_s=time_bin_edges,
        global_start_time_s=sensor_electric_fields["global_start_time_s"],
        lnb_power_min_fraction_of_max=1e-3,
        norm=None,
        lnb_power_min_W=0.0,
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

for t in range(noise_num_time_slices - numT):
    w = np.sum(total_power_leaving_lnb[:, t : t + numT, :], axis=1)
    total_power_sliding_integral[:, t, :] = w * simulation_time_slice_duration

simulation_time_slices_which_are_sampled_by_readout = np.arange(
    0, noise_num_time_slices, numT,
)
random_offset_or_readout_wrt_global_time_num_time_slices = int(
    prng.uniform(low=0, high=numT)
)
num_readout_frames = (
    len(simulation_time_slices_which_are_sampled_by_readout) - 1
)
readout_energy = np.zeros(
    shape=(sensor_electric_fields["num_antennas"], num_readout_frames, 2)
)
readout_global_start_time_s = (
    random_offset_or_readout_wrt_global_time_num_time_slices
    * timing["electric_fields"]["time_slice_duration_s"]
    + lnb_simulation_global_start_time_s
)
for i in range(num_readout_frames):
    simulation_time_slice = simulation_time_slices_which_are_sampled_by_readout[
        i
    ]
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
        signal=_Ain, time_slice_duration=1 / numT, window_num_slices=numT,
    )
    _r = np.sum(_Aout ** 2) / np.sum(_Ain ** 2)
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

# plot images seen by trigger
# ---------------------------
trigger_energy = iaat.telescope.apply_pixel_summation(
    signal=readout_energy,
    pixel_summation=telescope["trigger"]["pixel_summation"],
)

for units in ["electron_volt", "black_body_temperature", "jansky"]:
    plot_trigger_dir = os.path.join(plot_dir, "trigger", units)
    if not os.path.exists(plot_trigger_dir):
        iaat_plot.save_image_slices_energy_deposite(
            readout_energy_J=trigger_energy,
            readout_time_slice_duration_s=timing["readout"][
                "time_slice_duration_s"
            ],
            antenna_positions=np.rad2deg(
                telescope["sensor"]["feed_horn_positions_m"]
                / telescope["mirror"]["focal_length_m"]
            ),
            path=plot_trigger_dir,
            global_start_time_s=readout_global_start_time_s,
            units=units,
            bandwidth_Hz=telescope["lnb"]["intermediate_bandwidth_Hz"],
            mirror_area_m2=telescope["mirror"]["area_m2"],
            image_x_label="$c_x$ / (1$^{\circ}$)",
            image_y_label="$c_y$ / (1$^{\circ}$)",
        )
