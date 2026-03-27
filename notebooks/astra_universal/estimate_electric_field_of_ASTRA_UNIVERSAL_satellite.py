"""
How strong is the electric field seen by a comercial lnb of the ASTRA-universal
satellite?
"""

import numpy as np

# Estimate the max. areal power density ASTRA creates on earth
vacuum_impedance_ohm = 120 * np.pi
speed_of_light_m_per_s = 299792458.0

astra2e_local_oscillator_frequency_Hz = 9.75e9
astra2e_wavelength_m = (
    speed_of_light_m_per_s / astra2e_local_oscillator_frequency_Hz
)
astra2e_antenna_effective_area_m2 = (astra2e_wavelength_m**2) / (4.0 * np.pi)

astra2e_spots_m2 = {}
astra2e_spots_m2["kaband"] = (
    (np.pi / 4) * 3236e3 * 1640e3
)  # from astra2e_kaband.svg
astra2e_spots_m2["efg-pe"] = (
    (np.pi / 4) * 3569e3 * 2890e3
)  # from astra2e_efg-pe.svg
astra2e_spots_m2["me"] = (np.pi / 4) * 3980e3 * 3543e3  # from astra2e_me.svg
astra2e_spots_m2["efg-ukspot"] = (
    (np.pi / 4) * 2024e3 * 1415e3
)  # from astra2efg-ukspot.svg

astrea2e_area_m2 = np.sum([astra2e_spots_m2[k] for k in astra2e_spots_m2])
astra2e_max_power_W = 13.0e3  # from astra2e_satellite.md

astra2e_power_density_on_surface_of_earth_W_per_m2 = (
    astra2e_max_power_W / astrea2e_area_m2
)

typical_satellite_dish_area_m2 = 0.4**2 * np.pi
feedhorn_transmission = 1.0  # <- unrealistic, more typical value is 0.5.

power_arriving_in_lnb_W = (
    astra2e_power_density_on_surface_of_earth_W_per_m2
    * feedhorn_transmission
    * typical_satellite_dish_area_m2
)

electric_field_V_per_m = np.sqrt(
    (power_arriving_in_lnb_W * vacuum_impedance_ohm)
    / astra2e_antenna_effective_area_m2
)
# ohm = (V ** 2 / W)


print(
    "The ASTRA-lnb installed in a typical 80cm dish experiences "
    "an electric field of: "
    "{:.2f}uVm^{{-1}}".format(1e6 * electric_field_V_per_m)
)
