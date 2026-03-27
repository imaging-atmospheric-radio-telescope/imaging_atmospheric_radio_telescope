import imaging_atmospheric_radio_telescope as iart
import numpy as np


def test_power_spectrum():
    frequencies_Hz = np.linspace(1e9, 4e9, 50)

    time_slice_duration = 0.1e-9
    num_time_slices = 1360

    for frequency_Hz in frequencies_Hz:
        _, A = iart.signal.make_sin(
            frequency=frequency_Hz,
            time_slice_duration=time_slice_duration,
            num_time_slices=num_time_slices,
        )

        ffs_Hz, pds = iart.signal.estimate_power_spectrum_density(
            amplitudes=A,
            time_slice_duration_s=time_slice_duration,
            num_time_slices_to_average_over=num_time_slices // 5,
        )

        best_frequency_Hz = ffs_Hz[np.argmax(pds)]

        assert np.abs(best_frequency_Hz - frequency_Hz) < 100e6
