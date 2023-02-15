import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import numpy as np
import os

prng = np.random.Generator(np.random.PCG64(42))

lnb = iaat.lownoiseblock.ASTRA_UNIVERSAL

timing = iaat.timing_and_sampling.make_timing_from_lnb(
    lnb=lnb, oversampling=6, time_window_duration=35e-9,
)

mirror = iaat.telescope.make_mirror(
    random_seed=0,
    focal_length=18.9,
    outer_radius=6.3,
    inner_radius=3.15,
    probe_areal_density=14,
)

sensor = iaat.telescope.make_sensor(
    sensor_outer_radius=mirror["focal_length"] * np.deg2rad(3.25),
    sensor_distance=mirror["focal_length"],
    feed_horn_inner_radius=0.0274,
)

telescope = iaat.telescope.make_telescope(
    sensor=sensor,
    mirror=mirror,
    lnb=lnb,
    speed_of_light=iaat.signal.SPEED_OF_LIGHT,
)

telescope["transmission_from_air_into_feed_horn"] = 0.5

corsika_coreas_executable_path = os.path.join(
    "build", "corsika-77100", "run", "corsika77100Linux_QGSII_urqmd_coreas",
)

event_id = 201
event_path = "test{:06d}".format(event_id)

primary_particle = {
    "type": "gamma",
    "energy_GeV": 5000,
    "zenith_distance_rad": np.deg2rad(1.2),
    "azimuth_rad": 0.0,
    "core_north_m": 20,
    "core_west_m": 40,
}

"""
earth_radius_m = 6300e3
astra_power_W = 3.3e3
astra_earth_area_m2 = np.pi * earth_radius_m ** 2
astra_power_density_W_per_m2 = astra_power_W / astra_earth_area_m2
"""

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

"""
plot electic fields
"""
for component in ["probe", "mirror", "sensor"]:
    fig_path = os.path.join(event_path, "plot", component + ".jpg")
    if not os.path.exists(fig_path):
        field_path = os.path.join(event_path, component, "electric_fields.tar")
        field = iaat.electric_fields.read_tar(field_path)
        iaat_plot.write_figure_electric_fields_overview(
            electric_fields=field,
            path=fig_path
        )


sensor_electric_fields = iaat.electric_fields.read_tar(
    path=os.path.join(event_path, "sensor", "electric_fields.tar")
)
signal_efield_at_lnb = (
    feed_horn_gain
    * telescope["transmission_from_air_into_feed_horn"]
    * sensor_electric_fields["electric_fields"]
)

"""
mixer
"""
signal_efield_at_lnb = iaat.signal.lnb_mixer(
    amplitudes=signal_efield_at_lnb,
    time_slice_duration=timing["electric_fields"]["time_slice_duration"],
    local_oscillator_frequency=timing["lnb"]["local_oscillator_frequency"],
    intermediate_frequency_start=timing["lnb"]["intermediate_frequency_start"],
    intermediate_frequency_stop=timing["lnb"]["intermediate_frequency_stop"],
)

_signal_power = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area"],
    electric_field=signal_efield_at_lnb,
)

electric_field_thermal_noise_amplitude = iaat.signal.electric_field_of_thermal_noise(
    antenna_temperature_K=telescope["lnb"]["noise_temperature"],
    antenna_bandwidth=timing["lnb"]["bandwidth"],
)

noise_efield_at_lnb = np.sqrt(
    1 / telescope["lnb"]["effective_area"]
) * prng.normal(
    loc=0.0,
    scale=electric_field_thermal_noise_amplitude,
    size=(
        sensor_electric_fields["num_antennas"],
        sensor_electric_fields["num_time_slices"],
        3,
    ),
)

_noise_power = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area"],
    electric_field=noise_efield_at_lnb,
)

_expected_noise_power = iaat.signal.electric_power_of_thermal_noise(
    antenna_temperature_K=telescope["lnb"]["noise_temperature"],
    antenna_bandwidth=timing["lnb"]["bandwidth"],
)

total_efield_at_lnb = signal_efield_at_lnb + noise_efield_at_lnb

total_power = iaat.signal.calculate_antenna_power(
    effective_area=telescope["lnb"]["effective_area"],
    electric_field=total_efield_at_lnb,
)

total_power_integral = np.zeros(shape=total_power.shape)
T = 40 * timing["oversampling"]
for t in range(int(sensor_electric_fields["num_time_slices"]) - T):
    w = np.sum(total_power[:, t : t + T, :], axis=1)
    total_power_integral[:, t, :] = w


iaat_plot.save_image_slices_power(
    power=total_power_integral,
    time_slice_duration=timing["electric_fields"]["time_slice_duration"],
    antenna_positions=telescope["sensor"]["antenna_positions"],
    path=os.path.join(event_path, "plot", "sensor_noise"),
    time_slice_region_of_interest=np.arange(
        0, timing["electric_fields"]["sensor"]["num_time_slices"], T,
    ),
    dpi=80,
    figsize=(12, 4),
)
