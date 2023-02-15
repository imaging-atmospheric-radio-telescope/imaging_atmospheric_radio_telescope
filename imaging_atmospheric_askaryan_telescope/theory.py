import numpy as np


def allan_formula(
    energy_GeV=1e3,
    axis_distance_m=100,
    zenith_distance_rad=np.deg2rad(30),
    geomagnetic_angle_rad=np.deg2rad(30),
):
    R0 = 100.0

    energy_EeV = 1e-6 * energy_GeV
    field_uV_per_m_per_MHz = (
        20.0
        * (energy_EeV / 1e17)
        * np.sin(geomagnetic_angle_rad)
        * np.cos(zenith_distance_rad)
        * np.exp(-axis_distance_m / R0)
    )
    return field_uV_per_m_per_MHz


def airy_angle(mirror_diameter, wavelength):
    theta = np.arcsin(1.22 * wavelength / mirror_diameter)
    return theta
