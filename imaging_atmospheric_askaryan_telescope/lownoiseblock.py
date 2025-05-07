from . import signal
from . import time_series
from . import utils

import numpy as np


def init(key):
    if key == "astra_universal":
        lnb = {
            "key": key,
            "local_oscillator_frequency_Hz": 9.75e9,
            "local_oscillator_frequency_std_Hz": 1e6,
            "intermediate_frequency_start_Hz": 950e6,
            "intermediate_frequency_stop_Hz": 1950e6,
            "noise_temperature_K": 100,
        }
    elif key == "norsat_8215f_c_band":
        lnb = {
            "key": key,
            "local_oscillator_frequency_Hz": 5.15e9,
            "local_oscillator_frequency_std_Hz": 250e3,
            "intermediate_frequency_start_Hz": 950e6,
            "intermediate_frequency_stop_Hz": 1750e6,
            "noise_temperature_K": 45,
        }
    elif key == "inverto_40mm_pro_wideband":
        # FTA Communication Technologies S.a.r.l
        # IDLP-WDB02-OOPRO-OPP
        lnb = {
            "key": key,
            "local_oscillator_frequency_Hz": 10.41e9,
            "local_oscillator_frequency_std_Hz": 1e6,
            "intermediate_frequency_start_Hz": 230e6,
            "intermediate_frequency_stop_Hz": 2350e6,
            "noise_temperature_K": 100,
        }
    else:
        raise AttributeError(f"Lnb key '{key:s}' is not known.")

    lnb["effective_area_m2"] = signal.calculate_antenna_effective_area(
        wavelength=signal.frequency_to_wavelength(
            lnb["local_oscillator_frequency_Hz"]
        ),
        gain=1.0,
    )
    lnb["intermediate_bandwidth_Hz"] = (
        lnb["intermediate_frequency_stop_Hz"]
        - lnb["intermediate_frequency_start_Hz"]
    )
    lnb["noise_power_W"] = signal.electric_power_of_thermal_noise(
        antenna_temperature_K=lnb["noise_temperature_K"],
        antenna_bandwidth_Hz=lnb["intermediate_bandwidth_Hz"],
    )
    return lnb


def simulate_mixer(
    lnb_input_electric_fields,
    local_oscillator_frequency_Hz,
    local_oscillator_frequency_std_Hz,
    intermediate_frequency_start_Hz,
    intermediate_frequency_stop_Hz,
    random_seed,
):
    prng = np.random.Generator(np.random.PCG64(random_seed))

    E_input = lnb_input_electric_fields
    E_inter = time_series.zeros_like(other=E_input)
    E_output = time_series.zeros_like(other=E_input)

    TAU = 2.0 * np.pi
    for channel in range(E_input.num_channels):
        channel_random_phase = prng.uniform(low=0.0, high=TAU)
        channel_local_oscillator_frequency = utils.normal_approximation(
            prng=prng,
            mean=local_oscillator_frequency_Hz,
            std=local_oscillator_frequency_std_Hz,
            size=1,
            irwin_hall_order=3,
        )

        _, channel_sine_ampl = signal.make_sin(
            frequency=channel_local_oscillator_frequency,
            phase=channel_random_phase,
            time_slice_duration=E_input.time_slice_duration_s,
            num_time_slices=E_input.num_time_slices,
        )
        channel_sine_ampl = channel_sine_ampl.astype(E_input.dtype)

        for dim in range(E_input.num_components):
            ss = channel_sine_ampl * E_input[channel, :, dim]
            E_inter[channel, :, dim] = ss

    E_output = time_series.zeros_like(other=E_input)
    for channel in range(E_input.num_channels):
        for dim in range(E_input.num_components):
            ss = signal.butter_bandpass_filter(
                amplitudes=E_inter[channel, :, dim],
                frequency_start=intermediate_frequency_start_Hz,
                frequency_stop=intermediate_frequency_stop_Hz,
                time_slice_duration=E_input.time_slice_duration_s,
            )
            E_output[channel, :, dim] = ss

    assert E_input.dtype == E_output.dtype
    return E_output
