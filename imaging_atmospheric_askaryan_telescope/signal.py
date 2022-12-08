import scipy
from scipy.signal import butter
from scipy.signal import lfilter
import numpy as np


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(
    amplitudes,
    frequency_start,
    frequency_stop,
    time_slice_duration,
    order=5,
    axis=0,
):
    fs = 1.0 / time_slice_duration
    b, a = _butter_bandpass(
        lowcut=frequency_start, highcut=frequency_stop, fs=fs, order=order,
    )
    y = scipy.signal.lfilter(b, a, amplitudes, axis=axis)
    return y


def make_sin(frequency, time_slice_duration, num_time_slices):
    N = num_time_slices
    dt = time_slice_duration
    t = np.linspace(0, N * dt, N, endpoint=False)
    return t, np.sin(t * frequency * (2.0 * np.pi))


def butter_bench(
    frequencies=np.geomspace(0.1e9, 10e9, 100),
    bandpass=butter_bandpass_filter,
    filter_config={"frequency_start": 1e9, "frequency_stop": 2e9},
    num_time_slices=1000,
    time_slice_duration=0.5 * (1 / 10e9),
):
    ratio = []
    for f in frequencies:
        Ain = make_sin(
            frequency=f,
            time_slice_duration=time_slice_duration,
            num_time_slices=num_time_slices,
        )

        Aout = bandpass(
            amplitudes=Ain,
            **filter_config,
            time_slice_duration=time_slice_duration
        )

        r = np.sum(Aout ** 2) / np.sum(Ain ** 2)
        ratio.append(r)
    return ratio


VACUUM_IMPEDANCE = 120 * np.pi
SPEED_OF_LIGHT = 299792458.0
BOLTZMANN_CONSTANT = 1.38e-23


def frequency_to_wavelength(frequency, speed_of_light=SPEED_OF_LIGHT):
    return speed_of_light / frequency


def calculate_antenna_effective_area(wavelength, gain):
    return gain * ((wavelength ** 2) / (4.0 * np.pi))


def calculate_antenna_power(effective_area, electric_field):
    S = (electric_field ** 2) / VACUUM_IMPEDANCE
    Pr = effective_area * S
    return Pr


def electric_power_of_thermal_noise(
    antenna_temperature_K=80, antenna_bandwidth=1e9,
):
    return antenna_temperature_K * BOLTZMANN_CONSTANT * antenna_bandwidth


def electric_field_of_thermal_noise(
    antenna_temperature_K=80, antenna_bandwidth=1e9,
):
    P = electric_power_of_thermal_noise(
        antenna_temperature_K=antenna_temperature_K,
        antenna_bandwidth=antenna_bandwidth,
    )
    return np.sqrt(P * VACUUM_IMPEDANCE)
