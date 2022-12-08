import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import numpy as np
import os

prng = np.random.Generator(np.random.PCG64(42))

F_LNB_LOCAL_OSCILLATOR = 9.75e9
F_LNB_BLOCK_START = 10.7e9
F_LNB_BLOCK_STOP = 11.7e9
LNB_BANDWIDTH = F_LNB_BLOCK_STOP - F_LNB_BLOCK_START

F_OVERHEAD = 6
F_SIMULATION = F_OVERHEAD * F_LNB_LOCAL_OSCILLATOR
TIME_SLICE_DURATION = 1.0 / F_SIMULATION

NUM_TIME_SLICES = 4000

mirror = iaat.telescope.make_mirror(
    random_seed=0, focal_length=25.5, radius=8.5, probe_areal_density=6,
)

sensor = iaat.telescope.make_sensor(
    sensor_outer_radius=1.1, sensor_distance=25.5, feed_horn_inner_radius=0.03,
)

telescope = iaat.telescope.make_telescope(
    sensor=sensor, mirror=mirror, speed_of_light=3e8,
)

corsika_coreas_executable_path = os.path.join(
    "corsika_coreas_build",
    "corsika-77100",
    "run",
    "corsika77100Linux_QGSII_urqmd_coreas",
)

event_id = 127
event_path = "test{:06d}".format(event_id)

primary_particle = {
    "id": 1,
    "energy_GeV": 1000,
    "zenith_distance_rad": 0.0,
    "azimuth_rad": 0.0,
    "core_north_m": 10,
    "core_west_m": 50,
}
"""
astra_power = 3.3e3
astra_earth_area = np.pi * 6.3e3 ** 2
astra_power_density = astra_power / astra_earth_area
"""

lnb_effective_area = iaat.signal.calculate_antenna_effective_area(
    wavelength=iaat.signal.frequency_to_wavelength(F_LNB_LOCAL_OSCILLATOR),
    gain=0.5,
)

feed_horn_gain = sensor["antenna_area"] / lnb_effective_area

iaat.production.simulate_telescope_response(
    corsika_coreas_executable_path=corsika_coreas_executable_path,
    out_dir=event_path,
    event_id=event_id,
    primary_particle=primary_particle,
    site=iaat.sites.NAMIBIA,
    time_slice_duration=TIME_SLICE_DURATION,
    time_window_duration=400e-9,
    telescope=telescope,
    num_time_slices=NUM_TIME_SLICES,
)

sensor_electric_fields = iaat.telescope.read_electric_fields(
    path=os.path.join(event_path, "sensor", "electric_fields")
)
signal_efield_at_lnb = (
    feed_horn_gain * sensor_electric_fields["electric_fields"]
)

"""
mixer
"""
sine_time, sine_ampl = iaat.signal.make_sin(
    frequency=F_LNB_LOCAL_OSCILLATOR,
    time_slice_duration=TIME_SLICE_DURATION,
    num_time_slices=NUM_TIME_SLICES,
)
signal_mix_efield_at_lnb = np.zeros(shape=signal_efield_at_lnb.shape)
for channel in range(telescope["sensor"]["num_antennas"]):
    for dim in range(3):
        print(channel, dim)
        ss = sine_ampl * signal_efield_at_lnb[channel, :, dim]
        signal_mix_efield_at_lnb[channel, :, dim] = ss


signal_band_efield_at_lnb = np.zeros(shape=signal_efield_at_lnb.shape)
for channel in range(telescope["sensor"]["num_antennas"]):
    for dim in range(3):
        print(channel, dim)
        ss = iaat.signal.butter_bandpass_filter(
            amplitudes=signal_mix_efield_at_lnb[channel, :, dim],
            frequency_start=F_LNB_BLOCK_START - F_LNB_LOCAL_OSCILLATOR,
            frequency_stop=F_LNB_BLOCK_STOP - F_LNB_LOCAL_OSCILLATOR,
            time_slice_duration=sensor_electric_fields["time_slice_duration"],
        )
        signal_band_efield_at_lnb[channel, :, dim] = ss

signal_efield_at_lnb = signal_band_efield_at_lnb

_signal_power = iaat.signal.calculate_antenna_power(
    effective_area=lnb_effective_area, electric_field=signal_efield_at_lnb,
)

electric_field_thermal_noise_amplitude = iaat.signal.electric_field_of_thermal_noise(
    antenna_temperature_K=80, antenna_bandwidth=LNB_BANDWIDTH,
)

noise_efield_at_lnb = np.sqrt(1 / lnb_effective_area) * prng.normal(
    loc=0.0,
    scale=electric_field_thermal_noise_amplitude,
    size=(
        sensor_electric_fields["num_antennas"],
        sensor_electric_fields["num_time_slices"],
        3,
    ),
)

_noise_power = iaat.signal.calculate_antenna_power(
    effective_area=lnb_effective_area, electric_field=noise_efield_at_lnb,
)

_expected_noise_power = iaat.signal.electric_power_of_thermal_noise(
    antenna_temperature_K=80, antenna_bandwidth=1e9,
)

total_efield_at_lnb = signal_efield_at_lnb + noise_efield_at_lnb

total_power = iaat.signal.calculate_antenna_power(
    effective_area=lnb_effective_area, electric_field=total_efield_at_lnb,
)


total_power_integral = np.zeros(shape=total_power.shape)
T = 40 * F_OVERHEAD
for t in range(int(sensor_electric_fields["num_time_slices"]) - T):
    w = np.sum(total_power[:, t : t + T, :], axis=1)
    total_power_integral[:, t, :] = w


sensor_electric_fields["electric_fields"] = total_power_integral


iaat_plot.save_image_slices_electric_field(
    electric_fields=sensor_electric_fields,
    antenna_positions=telescope["sensor"]["antenna_positions"],
    path=os.path.join(event_path, "plot", "sensor_noise"),
    time_slice_region_of_interest=np.arange(0, NUM_TIME_SLICES, T),
    dpi=80,
    figsize=(12, 4),
)
