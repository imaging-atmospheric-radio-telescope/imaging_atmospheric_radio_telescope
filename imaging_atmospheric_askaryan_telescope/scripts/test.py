import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import numpy as np
import json_numpy
import os

# 58 ->    1 GHz
# 117 -> 500 MHz
# 234 -> 250 MHz
# 468 -> 125 MHz
# 702 -> 83.33 MHz

config = {
    "timing": {
        "oversampling": 6,
        "time_window_duration": 35e-9,
        "readout_integrates_num_simulation_time_slices": 234,
    },
    "mirror": {
        "random_seed": 0,
        "focal_length": 18.9,
        "outer_radius": 6.3,
        "inner_radius": 3.15,
        "probe_areal_density": 14,
    },
    "sensor": {
        "sensor_outer_radius": 18.9 * np.deg2rad(3.25),
        "sensor_distance": 18.9,
        "feed_horn_inner_radius": 0.0274,
    },
    "transmission_from_air_into_feed_horn": 0.5,
}

with open("config.json", "wt") as f:
    f.write(json_numpy.dumps(config))
with open("config.json", "rt") as f:
    config = json_numpy.loads(f.read())


lnb = iaat.lownoiseblock.ASTRA_UNIVERSAL
timing = iaat.timing_and_sampling.make_timing_from_lnb(
    lnb=lnb, **config["timing"],
)
mirror = iaat.telescope.make_mirror(**config["mirror"])
sensor = iaat.telescope.make_sensor(**config["sensor"])
telescope = iaat.telescope.make_telescope(
    sensor=sensor,
    mirror=mirror,
    lnb=lnb,
    speed_of_light=iaat.signal.SPEED_OF_LIGHT,
)
telescope["transmission_from_air_into_feed_horn"] = config[
    "transmission_from_air_into_feed_horn"
]

# start simulation
# ----------------
corsika_coreas_executable_path = os.path.join(
    "build", "corsika-77100", "run", "corsika77100Linux_QGSII_urqmd_coreas",
)

event_id = 205
prng = np.random.Generator(np.random.PCG64(event_id))

event_path = "test{:06d}".format(event_id)

primary_particle = {
    "type": "gamma",
    "energy_GeV": 2000,
    "zenith_distance_rad": np.deg2rad(0.5),
    "azimuth_rad": 30.0,
    "core_north_m": 20,
    "core_west_m": 5,
}

feed_horn_gain = (
    telescope["sensor"]["antenna_area"] / telescope["lnb"]["effective_area"]
)

iaat.production.simulate_telescope_response(
    corsika_coreas_executable_path=corsika_coreas_executable_path,
    out_dir=event_path,
    event_id=event_id,
    primary_particle=primary_particle,
    site=iaat.sites.NAMIBIA,
    telescope=telescope,
    timing=timing,
)

# plot electic fields
# -------------------
plot_dir = os.path.join(event_path, "plot")
os.makedirs(plot_dir, exist_ok=True)
for component in ["probe", "mirror", "sensor"]:
    fig_path = os.path.join(plot_dir, component + ".jpg")
    if not os.path.exists(fig_path):
        field_path = os.path.join(event_path, component, "electric_fields.tar")
        field = iaat.electric_fields.read_tar(field_path)
        iaat_plot.write_figure_electric_fields_overview(
            electric_fields=field, path=fig_path, component_mask=[1, 1, 0],
        )

    fig_spectrum_path = os.path.join(
        plot_dir, component + "_power_spectrum_density.jpg"
    )
    if True:  # not os.path.exists(fig_spectrum_path):
        field_path = os.path.join(event_path, component, "electric_fields.tar")
        field = iaat.electric_fields.read_tar(field_path)
        iaat_plot.write_figure_electric_fields_power_density_spectrum(
            path=fig_spectrum_path,
            electric_fields=field,
            component_mask=[1, 1, 0],
            num_time_slices_to_average_over=(
                field["electric_fields"].shape[1] // 20
            ),
        )

# plot instrument
# ---------------
for component in ["mirror", "sensor"]:
    fig_path = os.path.join(plot_dir, component + ".antenna_positions.jpg")
    if not os.path.exists(fig_path):
        iaat_plot.write_figure_antenna_positions(
            positions=telescope[component]["antenna_positions"], path=fig_path
        )

sensor_electric_fields = iaat.electric_fields.read_tar(
    path=os.path.join(event_path, "sensor", "electric_fields.tar")
)
signal_efield_in_lnb = (
    feed_horn_gain
    * telescope["transmission_from_air_into_feed_horn"]
    * sensor_electric_fields["electric_fields"]
)

# lnb mixer
# ---------
signal_efield_leaving_lnb = iaat.signal.lnb_mixer(
    amplitudes=signal_efield_in_lnb,
    time_slice_duration=timing["electric_fields"]["time_slice_duration"],
    local_oscillator_frequency=telescope["lnb"]["local_oscillator_frequency"],
    intermediate_frequency_start=telescope["lnb"][
        "intermediate_frequency_start"
    ],
    intermediate_frequency_stop=telescope["lnb"][
        "intermediate_frequency_stop"
    ],
)

# plot lnb mixer gain
# -------------------
_lnb_bench_frequency = np.geomspace(0.1e9, 10e9, 100)
_lnb_bench_gain = iaat.signal.butter_bench(
    frequencies=_lnb_bench_frequency,
    bandpass=iaat.signal.butter_bandpass_filter,
    filter_config={
        "frequency_start": telescope["lnb"]["intermediate_frequency_start"],
        "frequency_stop": telescope["lnb"]["intermediate_frequency_stop"],
    },
    num_time_slices=10000,
    time_slice_duration=timing["electric_fields"]["time_slice_duration"],
)
fig_path_lnb_gain = os.path.join(plot_dir, "lnb_gain.jpg")
if not os.path.exists(fig_path_lnb_gain):
    iaat.plot.write_figure_gain(
        path=fig_path_lnb_gain,
        frequency=_lnb_bench_frequency,
        gain=_lnb_bench_gain,
    )

# power
# -----
_signal_power = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area"],
    electric_field=signal_efield_leaving_lnb,
)

electric_field_thermal_noise_amplitude = iaat.signal.electric_field_of_thermal_noise(
    antenna_temperature_K=telescope["lnb"]["noise_temperature"],
    antenna_bandwidth=telescope["lnb"]["intermediate_bandwidth"],
)

noise_num_time_slices = int(sensor_electric_fields["num_time_slices"]) * int(2)
lnb_simulation_global_start_time = (
    sensor_electric_fields["global_start_time"]
    - sensor_electric_fields["num_time_slices"]
    * sensor_electric_fields["time_slice_duration"]
)

noise_efield_leaving_lnb = np.sqrt(
    1 / telescope["lnb"]["effective_area"]
) * prng.normal(
    loc=0.0,
    scale=electric_field_thermal_noise_amplitude,
    size=(sensor_electric_fields["num_antennas"], noise_num_time_slices, 3,),
)

_noise_power = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area"],
    electric_field=noise_efield_leaving_lnb,
)

_expected_noise_power = iaat.signal.electric_power_of_thermal_noise(
    antenna_temperature_K=telescope["lnb"]["noise_temperature"],
    antenna_bandwidth=telescope["lnb"]["intermediate_bandwidth"],
)

assert 0.9 < (_expected_noise_power / np.mean(_noise_power)) < 1.1

numS = sensor_electric_fields["num_time_slices"]
total_efield_leaving_lnb = noise_efield_leaving_lnb
total_efield_leaving_lnb[:, numS:, :] += signal_efield_leaving_lnb

total_power_leaving_lnb = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area"],
    electric_field=total_efield_leaving_lnb,
)

# plot power_leaving_lnb
# ----------------------
fig_path_power_leaving_lnb = os.path.join(plot_dir, "lnb_output.jpg")
if not os.path.exists(fig_path_power_leaving_lnb):
    total_power_leaving_lnb_xy = np.sum(
        total_power_leaving_lnb[:, numS:, 0:2], axis=2,
    )
    antenna_bin_edges = iaat.electric_fields.make_antenna_bin_edges(
        electric_fields=sensor_electric_fields,
    )
    time_bin_edges = iaat.electric_fields.make_time_bin_edges(
        electric_fields=sensor_electric_fields, global_time=False,
    )
    pmax_pW = 1e12 * np.max(total_power_leaving_lnb_xy)
    iaat_plot.write_figure_lnb_power(
        path=fig_path_power_leaving_lnb,
        lnb_power=total_power_leaving_lnb_xy,
        antenna_bin_edges=antenna_bin_edges,
        relative_time_bin_edges=time_bin_edges,
        global_start_time=sensor_electric_fields["global_start_time"],
        vim_fraction_of_vmax=1e-3,
        vmax=pmax_pW,
        vmin=0.5 * 1e12 * _expected_noise_power,
        norm=iaat_plot.matplotlib.colors.LogNorm(),
        dpi=400,
        expected_noise_power=_expected_noise_power,
    )

# integrate over time for readout
# -------------------------------
total_power_sliding_integral = np.zeros(shape=total_power_leaving_lnb.shape)

numT = timing["readout"]["integrates_num_simulation_time_slices"]
simulation_time_slice_duration = timing["electric_fields"][
    "time_slice_duration"
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
readout_global_start_time = (
    random_offset_or_readout_wrt_global_time_num_time_slices
    * timing["electric_fields"]["time_slice_duration"]
    + lnb_simulation_global_start_time
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


for units in ["electron_volt", "black_body_temperature", "jansky"]:
    plot_sensor_dir = os.path.join(plot_dir, "readout", units)
    if not os.path.exists(plot_sensor_dir):
        iaat_plot.save_image_slices_energy_deposite(
            readout_energy=readout_energy,
            readout_time_slice_duration=timing["readout"][
                "time_slice_duration"
            ],
            antenna_positions=telescope["sensor"]["antenna_positions"],
            path=plot_sensor_dir,
            global_start_time=readout_global_start_time,
            dpi=80,
            figsize=(12, 4),
            units=units,
            bandwidth=telescope["lnb"]["intermediate_bandwidth"],
            mirror_area=telescope["mirror"]["area"],
        )

trigger_energy = iaat.telescope.apply_pixel_summation(
    signal=readout_energy,
    pixel_summation=telescope["trigger"]["pixel_summation"],
)

for units in ["electron_volt", "black_body_temperature", "jansky"]:
    plot_trigger_dir = os.path.join(plot_dir, "trigger", units)
    if not os.path.exists(plot_trigger_dir):
        iaat_plot.save_image_slices_energy_deposite(
            readout_energy=trigger_energy,
            readout_time_slice_duration=timing["readout"][
                "time_slice_duration"
            ],
            antenna_positions=telescope["sensor"]["antenna_positions"],
            path=plot_trigger_dir,
            global_start_time=readout_global_start_time,
            dpi=80,
            figsize=(12, 4),
            units=units,
            bandwidth=telescope["lnb"]["intermediate_bandwidth"],
            mirror_area=telescope["mirror"]["area"],
        )
