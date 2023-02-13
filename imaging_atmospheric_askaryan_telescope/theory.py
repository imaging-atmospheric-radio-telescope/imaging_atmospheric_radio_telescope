import numpy as np


def allan_formula(
    energy_GeV,
    axis_distance_m,
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
