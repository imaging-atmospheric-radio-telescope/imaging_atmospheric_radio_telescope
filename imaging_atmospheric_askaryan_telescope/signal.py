import scipy
from scipy.signal import butter
from scipy.signal import lfilter
import numpy as np
import copy


def add_first_to_second_at(first, second, at):
    """
    Adds the values in 'first' to the values in 'second' with the
    starting index in 'second' being 'at'.

    Parameters
    ----------
    first : array
        A series of values to be added to 'second'.
    second : array
        A series of values to add values from 'first' to.
        'second' is modified in place.
    at : int
        The starting index in 'second' where the values of 'first' are added
        to.
    """
    if at > second.shape[0]:
        return

    end = at + first.shape[0]
    if end < 0:
        return

    if end >= second.shape[0]:
        end = second.shape[0]

    start = at
    if start < 0:
        start = 0

    second[start:end] += first[start - at : end - at]


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
    """
    Return amplitudes after passing a band-pass.

    Parameters
    ----------
    amplitudes : array (float)
        The amplitudes of a signal
    time_slice_duration : float
        Defines the duration of a slice in 'amplitudes'.
    frequency_start : float
        Lower frequency of band-pass.
    frequency_stop : float
        Upper frequency of band-pass.
    """
    fs = 1.0 / time_slice_duration
    b, a = _butter_bandpass(
        lowcut=frequency_start,
        highcut=frequency_stop,
        fs=fs,
        order=order,
    )
    y = scipy.signal.lfilter(b, a, amplitudes, axis=axis)
    return y


def make_sin(frequency, time_slice_duration, num_time_slices):
    """
    Returns the moments and amplitudes (2-tuple) of a sine-wave.
    """
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
    """
    A benchmark for band-pass-filters.
    It estimates the transmission-ratio for multiple frequencies.
    """
    ratio = []
    for f in frequencies:
        _t, Ain = make_sin(
            frequency=f,
            time_slice_duration=time_slice_duration,
            num_time_slices=num_time_slices,
        )

        Aout = bandpass(
            amplitudes=Ain,
            **filter_config,
            time_slice_duration=time_slice_duration,
        )

        r = np.sum(Aout**2) / np.sum(Ain**2)
        ratio.append(r)
    return ratio


VACUUM_IMPEDANCE_OHM = 120 * np.pi
SPEED_OF_LIGHT_M_PER_S = 299792458.0
BOLTZMANN_CONSTANT_J_PER_K = 1.38e-23
STANDARD_DEVIATION_OF_RECTANGULAR_FUNCTION = 1.0 / np.sqrt(12.0)


def frequency_to_wavelength(frequency, speed_of_light=SPEED_OF_LIGHT_M_PER_S):
    return speed_of_light / frequency


def calculate_antenna_effective_area(wavelength, gain):
    return gain * ((wavelength**2) / (4.0 * np.pi))


def calculate_antenna_power(effective_area, electric_field):
    S = (electric_field**2) / VACUUM_IMPEDANCE_OHM
    Pr = effective_area * S
    return Pr


def electric_power_of_thermal_noise(
    antenna_temperature_K=80,
    antenna_bandwidth_Hz=1e9,
):
    return (
        antenna_temperature_K
        * BOLTZMANN_CONSTANT_J_PER_K
        * antenna_bandwidth_Hz
    )


def radiated_power_to_blackbody_temperature(power_W, bandwidth_Hz):
    return power_W / (BOLTZMANN_CONSTANT_J_PER_K * bandwidth_Hz)


def electric_field_of_thermal_noise(
    antenna_temperature_K=80,
    antenna_bandwidth_Hz=1e9,
):
    P = electric_power_of_thermal_noise(
        antenna_temperature_K=antenna_temperature_K,
        antenna_bandwidth_Hz=antenna_bandwidth_Hz,
    )
    return np.sqrt(P * VACUUM_IMPEDANCE_OHM)


def lnb_mixer(
    amplitudes,
    time_slice_duration,
    local_oscillator_frequency,
    intermediate_frequency_start,
    intermediate_frequency_stop,
):
    num_channels, num_time_slices, num_dims = amplitudes.shape

    _, sine_ampl = make_sin(
        frequency=local_oscillator_frequency,
        time_slice_duration=time_slice_duration,
        num_time_slices=num_time_slices,
    )

    amplmix = np.zeros(shape=amplitudes.shape)
    for channel in range(num_channels):
        for dim in range(num_dims):
            ss = sine_ampl * amplitudes[channel, :, dim]
            amplmix[channel, :, dim] = ss

    amplban = np.zeros(shape=amplitudes.shape)
    for channel in range(num_channels):
        for dim in range(num_dims):
            ss = butter_bandpass_filter(
                amplitudes=amplmix[channel, :, dim],
                frequency_start=intermediate_frequency_start,
                frequency_stop=intermediate_frequency_stop,
                time_slice_duration=time_slice_duration,
            )
            amplban[channel, :, dim] = ss

    return amplban


def estimate_power_spectrum_density(
    amplitudes, time_slice_duration, num_time_slices_to_average_over
):
    sampling_frequency = 1.0 / time_slice_duration
    frequencies, power_density = scipy.signal.welch(
        x=amplitudes,
        fs=sampling_frequency,
        nperseg=num_time_slices_to_average_over,
        scaling="density",
        average="mean",
    )
    return frequencies, power_density


def integrate_sliding_window(signal, time_slice_duration, window_num_slices):
    T = window_num_slices
    signal_num_slices = signal.shape[0]
    out = np.zeros(signal_num_slices)
    for t in range(signal_num_slices - T):
        out[t] = np.sum(signal[t : t + T]) * time_slice_duration
    return out


def correlate_phase_angle_of_sine_wave(
    time_s,
    signal,
    sine_wave_frequency_Hz,
    phase_angles_rad=np.linspace(-np.pi, np.pi, 100),
):
    assert sine_wave_frequency_Hz > 0.0

    TAU = 2.0 * np.pi
    correlations = []
    for phase_shift_rad in phase_angles_rad:
        reference_signal = np.sin(
            time_s * sine_wave_frequency_Hz * TAU + phase_shift_rad
        )
        correlation = np.sum(np.abs(signal * reference_signal))
        correlations.append(correlation)

    correlations /= np.max(correlations)
    return np.array(correlations), np.array(phase_angles_rad)


def estimate_phase_angle_of_sine_wave(
    time_s,
    signal,
    sine_wave_frequency_Hz,
    max_num_iterations=10,
    precision_rad=1e-2,
):
    """
    Estimate the phase angle of a sine wave in 'signal'.

    Parameters
    ----------
    time_s : array like
        Time points where 'signal' is sampled.
    signal : array like
        Amplitude of the signal w.r.t. the time 'time_s'.
    sine_wave_frequency_Hz : float
        Frequency of the sine wave to look for.

    Returns
    -------
    phase_rad : float
        Estimated phase angle.
    """
    phase_rad = np.pi
    phase_radius_rad = np.pi
    num_iterations = 0
    assert max_num_iterations > 0
    assert precision_rad > 0

    while True:
        assert num_iterations <= max_num_iterations
        num_iterations += 1

        phase_angles_rad = np.linspace(
            phase_rad - phase_radius_rad, phase_rad + phase_radius_rad, 50
        )

        correlation, _ = correlate_phase_angle_of_sine_wave(
            time_s=time_s,
            signal=signal,
            sine_wave_frequency_Hz=sine_wave_frequency_Hz,
            phase_angles_rad=phase_angles_rad,
        )

        old_phase_rad = copy.copy(phase_rad)

        phase_rad = phase_angles_rad[np.argmax(correlation)]
        phase_radius_rad *= 0.5

        if np.abs(phase_rad - old_phase_rad) < precision_rad:
            break

    return phase_rad
