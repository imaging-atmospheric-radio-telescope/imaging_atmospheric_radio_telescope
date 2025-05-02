import numpy as np
import imaging_atmospheric_askaryan_telescope as iaat
import json_numpy
import math

# setup
# -----
config = iaat.from_config("run")
telescope = config["telescope"]
timing = config["timing"]

# start simulation
# ----------------
prng = np.random.Generator(np.random.PCG64(42))

electric_field_thermal_noise_amplitude_V_per_m = (
    iaat.signal.calculate_electric_field_strength_of_thermal_noise_V_per_m(
        antenna_temperature_K=telescope["lnb"]["noise_temperature_K"],
        antenna_bandwidth_Hz=telescope["lnb"]["intermediate_bandwidth_Hz"],
        antenna_effective_area_m2=telescope["lnb"]["effective_area_m2"],
    )
)

noise_num_time_slices = 1000 * 1000

num_neighbor_pixels_in_trigger = 7

numT = timing["readout"]["integrates_num_simulation_time_slices"]
simulation_time_slice_duration = timing["electric_fields"][
    "time_slice_duration_s"
]

simulation_time_slices_which_are_sampled_by_readout = np.arange(
    0,
    noise_num_time_slices,
    numT,
)
num_readout_frames = (
    len(simulation_time_slices_which_are_sampled_by_readout) - 1
)

readout_energy_blocks = []
for block in range(10):
    print(block)

    noise_efield_leaving_lnb = prng.normal(
        loc=0.0,
        scale=electric_field_thermal_noise_amplitude_V_per_m,
        size=noise_num_time_slices,
    )

    _noise_power = iaat.signal.calculate_antenna_power(
        effective_area=telescope["lnb"]["effective_area_m2"],
        electric_field=noise_efield_leaving_lnb,
    )

    assert (
        0.9 < (telescope["lnb"]["noise_power_W"] / np.mean(_noise_power)) < 1.1
    )

    # integrate over time for readout
    # -------------------------------
    noise_energy = np.zeros(shape=_noise_power.shape)

    for t in range(noise_num_time_slices - numT):
        w = np.sum(_noise_power[t : t + numT])
        noise_energy[t] = w * simulation_time_slice_duration

    _readout_energy = np.zeros(num_readout_frames)

    for i in range(num_readout_frames):
        simulation_time_slice = (
            simulation_time_slices_which_are_sampled_by_readout[i]
        )
        _readout_energy[i] = noise_energy[simulation_time_slice]

    expected_noise_energy_in_read_out_slice = (
        telescope["lnb"]["noise_power_W"]
        * timing["readout"]["time_slice_duration_s"]
    )

    readout_energy_blocks.append(_readout_energy)

readout_energy = np.concatenate(readout_energy_blocks)


two_components_x_and_y = 2
trigger_energy_std = (
    np.std(readout_energy)
    * (1 / np.sqrt(two_components_x_and_y))
    * (1 / np.sqrt(num_neighbor_pixels_in_trigger))
)
trigger_energy_mean = (
    np.mean(readout_energy)
    * two_components_x_and_y
    * num_neighbor_pixels_in_trigger
)


acceptable_accidental_trigger_rate_Hz = 1 / 3600
acceptable_time_unti_next_accidental_trigger_s = (
    1 / acceptable_accidental_trigger_rate_Hz
)
num_readout_time_slices_until_acceptable_accidental_trigger = (
    acceptable_time_unti_next_accidental_trigger_s
    / timing["readout"]["time_slice_duration_s"]
)
acceptable_probability_for_readout_time_slice_to_trigger_accidentally = (
    1.0 / num_readout_time_slices_until_acceptable_accidental_trigger
) * (1.0 / telescope["sensor"]["num_feed_horns"])

sigma = 8.1
p = 1 - math.erf(sigma / np.sqrt(2))

trigger_energy_threshold_J = trigger_energy_mean + sigma * trigger_energy_std
trigger_energy_threshold_eV = (
    trigger_energy_threshold_J / iaat.signal.ELECTRON_VOLT_J
)
trigger_energy_threshold_K = (
    iaat.signal.radiated_power_to_blackbody_temperature(
        power_W=trigger_energy_threshold_J
        / timing["readout"]["time_slice_duration_s"],
        bandwidth_Hz=telescope["lnb"]["intermediate_bandwidth_Hz"],
    )
)

print(
    "Trigger threshols: ",
    trigger_energy_threshold_eV,
    "eV, or ",
    trigger_energy_threshold_K,
    "K,  sigma=8.1, 1 accidental in 1h.",
)
