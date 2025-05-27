import numpy as np


def allan_formula(
    energy_GeV=1e3,
    axis_distance_m=100,
    zenith_distance_rad=np.deg2rad(30),
    geomagnetic_angle_rad=np.deg2rad(30),
):
    """
    Allans formula returns the expected electric field strngth per bandwith
    induced by an airshower.

    Parameters
    ----------
    energy_GeV : float
        Primary particle's energy.
    axis_distance_m : float
        Distance of observer to shower axis.
    zenith_distance_rad : float
        Zenith diastance of shower axis.
    geomagnetic_angle_rad : float

    Returns
    -------
    field_uV_per_m_per_MHz : float

    """
    R0 = 100.0

    energy_eV = 1e9 * energy_GeV
    field_uV_per_m_per_MHz = (
        20.0
        * (energy_eV / 1e17)
        * np.sin(geomagnetic_angle_rad)
        * np.cos(zenith_distance_rad)
        * np.exp(-axis_distance_m / R0)
    )
    return field_uV_per_m_per_MHz


def airy_angle(mirror_diameter, wavelength):
    """
    Airy's angle is the best possible angular resolution possible with a disk
    like aperture.

    Parameters
    ----------
    mirror_diameter : float
        Diameter of aperture.
    wavelength : float

    Returns
    -------
    theta : float
        Airy's angle in rad.
    """
    assert mirror_diameter > 0
    assert wavelength > 0

    theta = np.arcsin(1.22 * wavelength / mirror_diameter)
    return theta
