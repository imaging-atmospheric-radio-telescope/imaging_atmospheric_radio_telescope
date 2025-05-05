from . import signal
from . import time_series


def init(key):
    if key == "astra_universal":
        lnb = {
            "key": key,
            "local_oscillator_frequency_Hz": 9.75e9,
            "intermediate_frequency_start_Hz": 950e6,
            "intermediate_frequency_stop_Hz": 1950e6,
            "noise_temperature_K": 100,
        }
    elif key == "norsat_8215f_c_band":
        lnb = {
            "key": key,
            "local_oscillator_frequency_Hz": 5.15e9,
            "intermediate_frequency_start_Hz": 950e6,
            "intermediate_frequency_stop_Hz": 1750e6,
            "noise_temperature_K": 45,
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
    local_oscillator_frequency,
    intermediate_frequency_start,
    intermediate_frequency_stop,
):
    E_input = lnb_input_electric_fields
    E_inter = time_series.zeros_like(other=E_input)
    E_output = time_series.zeros_like(other=E_input)

    _, sine_ampl = signal.make_sin(
        frequency=local_oscillator_frequency,
        time_slice_duration=E_input.time_slice_duration_s,
        num_time_slices=E_input.num_time_slices,
        dtype=E_input.dtype,
    )
    sine_ampl = sine_ampl.astype(E_input.dtype)

    for channel in range(E_input.num_channels):
        for dim in range(E_input.num_components):
            ss = sine_ampl * E_input[channel, :, dim]
            E_inter[channel, :, dim] = ss

    E_output = time_series.zeros_like(other=E_input)
    for channel in range(E_input.num_channels):
        for dim in range(E_input.num_components):
            ss = signal.butter_bandpass_filter(
                amplitudes=E_inter[channel, :, dim],
                frequency_start=intermediate_frequency_start,
                frequency_stop=intermediate_frequency_stop,
                time_slice_duration=E_input.time_slice_duration_s,
            )
            E_output[channel, :, dim] = ss

    assert E_input.dtype == E_output.dtype
    return E_output
