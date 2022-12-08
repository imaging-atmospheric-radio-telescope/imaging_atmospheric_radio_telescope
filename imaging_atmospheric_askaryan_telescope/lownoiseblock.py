from . import signal


ASTRA_UNIVERSAL = {
    "local_oscillator_frequency": 9.75e9,
    "intermediate_frequency_start": 950e6,
    "intermediate_frequency_stop": 1950e6,
    "noise_temperature": 100,
}
ASTRA_UNIVERSAL["effective_area"] = signal.calculate_antenna_effective_area(
    wavelength=signal.frequency_to_wavelength(
        ASTRA_UNIVERSAL["local_oscillator_frequency"]
    ),
    gain=1.0,
)
ASTRA_UNIVERSAL["noise_power"] = signal.electric_power_of_thermal_noise(
    antenna_temperature_K=ASTRA_UNIVERSAL["noise_temperature"],
    antenna_bandwidth=(
        ASTRA_UNIVERSAL["intermediate_frequency_stop"]
        - ASTRA_UNIVERSAL["intermediate_frequency_start"]
    ),
)
