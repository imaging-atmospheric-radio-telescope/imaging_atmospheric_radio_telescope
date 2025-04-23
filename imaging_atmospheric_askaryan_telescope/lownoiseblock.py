from . import signal


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
