from .. import signal

import numpy as np


def compute_energy_freqband(E, dt, antenna_effective_area_m2, f_band=None):
    """
    Compute energy per pixel with optional Butterworth bandpass filter.

    Returns:
        feed_horn_energies_eV : array of shape (N_pix,)
    """
    if f_band is not None:
        E = signal.butter_bandpass_filter(
            amplitudes=E,
            frequency_start=f_band[0],
            frequency_stop=f_band[1],
            time_slice_duration=dt,
            axis=1,
        )

    E2 = E[:, :, 0] ** 2 + E[:, :, 1] ** 2 + E[:, :, 2] ** 2
    P_W = signal.calculate_antenna_power_W(
        effective_area_m2=antenna_effective_area_m2,
        electric_field_V_per_m=np.sqrt(E2),
    )
    Ene_J = np.sum(P_W, axis=1) * dt
    feed_horn_energies_eV = Ene_J / signal.ELECTRON_VOLT_J

    return feed_horn_energies_eV
