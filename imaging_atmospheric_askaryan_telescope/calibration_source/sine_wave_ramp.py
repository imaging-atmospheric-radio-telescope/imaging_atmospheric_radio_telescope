import numpy as np


def time_to_slice(t, dt):
    """
    Parameters
    ----------
    t : float
        Time.
    dt : float
        Time slice duration.

    Returns
    -------
    slice : int
    """
    assert dt > 0.0
    return int(np.round(t / dt))


def make_sine_wave_with_ramp_up_and_ramp_down(
    emission_frequency_Hz,
    emission_start_time_s,
    emission_duration_s,
    emission_ramp_up_duration_s,
    emission_ramp_down_duration_s,
    global_start_time_s,
    time_slice_duration_s,
    num_time_slices,
):
    """
     max. amplitude
     of sine wave
         ^
         |            emission_duration_s
         |           |------------------|
    1.0 _|_          ___________________
         |         /|                   | \
         |       /  |                   |   \
         |     /    |                   |     \
    0.0 -|---|------|-------------------|------|-----> time
             t_up   |                   t_down t_end
                    |
                    t_emission (emission_start_time_s)

    Parameters
    ----------
    emission_frequency_Hz : float
        Sine wave frequency.
    emission_start_time_s : float
        Begin of emission time with amplitude 1.0. The optional ramp up
        duration is before this start time.
    emission_duration_s : float
        Time duration of sine wave with amplitude 1.0. The duration starts
        at time 'emission_start_time_s'.
    emission_ramp_up_duration_s : float
        Duration of linear amplitude ramp up befor 'emission_start_time_s'.
    emission_ramp_down_duration_s : float
        Duration of linear amlitude ramp down after the 'emission_duration_s'
        is over.
    global_start_time_s : float
        The absolut time of the first time slice in the output array.
    time_slice_duration_s : float
        The output array will be sampled in time slices of this duration.
    num_time_slices : int
        The output array will be this many time slices long.

    Returns
    -------
    A : array of floats
        The amplitude of a sine wave with optinal ramp up and ramp down sampled
        in 'num_time_slices' equally long time slices of
        'time_slice_duration_s'.
    """
    MIN_OVERSAMPLNG_RATIO = 3.0
    assert emission_frequency_Hz > 0.0
    assert emission_duration_s >= 0.0
    assert emission_ramp_up_duration_s >= 0.0
    assert emission_ramp_down_duration_s >= 0.0
    assert time_slice_duration_s > 0.0
    assert num_time_slices >= 0
    assert (
        1.0 / time_slice_duration_s
        >= MIN_OVERSAMPLNG_RATIO * emission_frequency_Hz
    )

    N = num_time_slices
    dt = time_slice_duration_s

    # in time relative to 'global_start_time_s'
    t_start = emission_start_time_s - global_start_time_s
    t_up = t_start - emission_ramp_up_duration_s
    t_down = t_start + emission_duration_s
    t_end = t_down + emission_ramp_down_duration_s

    # in slices
    s_up = time_to_slice(t=t_up, dt=dt)
    s_start = time_to_slice(t=t_start, dt=dt)
    s_down = time_to_slice(t=t_down, dt=dt)
    s_end = time_to_slice(t=t_end, dt=dt)

    TAU = 2.0 * np.pi

    # init the time 't'
    t = np.linspace(0.0, N * dt, N, endpoint=False)
    t += global_start_time_s

    # init the amplitude 'A'
    A = np.sin((t - emission_start_time_s) * emission_frequency_Hz * TAU)

    # zeros before s_up
    # -----------------
    for s in np.arange(0, min([N, s_up])):
        if 0 <= s < N:
            A[s] = 0.0

    # ramp up
    # -------
    N_ramp_up = s_start - s_up
    for s in np.arange(s_up, s_start):
        weight = (s - s_up) / N_ramp_up
        if 0 <= s < N:
            A[s] = A[s] * weight

    # ramp down
    # ---------
    N_ramp_down = s_end - s_down
    for s in np.arange(s_down, s_end):
        weight = 1.0 - ((s - s_down) / N_ramp_down)
        if 0 <= s < N:
            A[s] = A[s] * weight

    # zeros after s_end
    # -----------------
    for s in np.arange(s_end, N):
        if 0 <= s < N:
            A[s] = 0.0

    return A
