from . import signal


def init(lnb_name):
    if lnb_name == "astra_universal":
        lnb = {
            "name": lnb_name,
            "local_oscillator_frequency": 9.75e9,
            "intermediate_frequency_start": 950e6,
            "intermediate_frequency_stop": 1950e6,
            "noise_temperature": 100,
        }
    elif lnb_name == "norsat_8215f_c_band":
        lnb = {
            "name": lnb_name,
            "local_oscillator_frequency": 5.15e9,
            "intermediate_frequency_start": 950e6,
            "intermediate_frequency_stop": 1750e6,
            "noise_temperature": 45,
        }
    else:
        raise AttributeError("lnb_name is not known.")

    lnb["effective_area"] = signal.calculate_antenna_effective_area(
        wavelength=signal.frequency_to_wavelength(
            lnb["local_oscillator_frequency"]
        ),
        gain=1.0,
    )
    lnb["intermediate_bandwidth"] = (
        lnb["intermediate_frequency_stop"]
        - lnb["intermediate_frequency_start"]
    )
    lnb["noise_power"] = signal.electric_power_of_thermal_noise(
        antenna_temperature_K=lnb["noise_temperature"],
        antenna_bandwidth=lnb["intermediate_bandwidth"],
    )
    return lnb
