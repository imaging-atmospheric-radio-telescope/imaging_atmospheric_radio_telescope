import numpy as np


def sample_zenith_distance(
    min_zenith_distance=np.deg2rad(1),
    max_zenith_distance=np.deg2rad(5),
    size=100,
):
    v_min = (np.cos(min_zenith_distance) + 1) / 2
    v_max = (np.cos(max_zenith_distance) + 1) / 2
    v = np.random.uniform(low=v_min, high=v_max, size=size)
    return np.arccos(2 * v - 1)


def sample_2D_points_within_radius(radius, size):
    rho = np.sqrt(np.random.uniform(0, 1, size)) * radius
    phi = np.random.uniform(0, 2 * np.pi, size)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def draw_event_parameters(
    number_events=960,
    primary_particle_id=1,
    energy=[100, 1000],
    azimuth=np.deg2rad([0.0, 360]),
    zenith_distance=np.deg2rad([0.0, 1.5]),
    observation_level_altitude=2200,
    core_position_on_observation_level_max_scatter_radius=100,
    time_slice_duration=2e-10,
):
    max_r = core_position_on_observation_level_max_scatter_radius

    core_north, core_west = sample_2D_points_within_radius(
        radius=core_position_on_observation_level_max_scatter_radius,
        size=number_events,
    )

    return {
        "event_id": np.arange(1, number_events + 1),
        "time_slice_duration": np.repeat(time_slice_duration, number_events),
        "primary_particle_id": np.repeat(primary_particle_id, number_events),
        "energy": np.random.uniform(
            low=energy[0], high=energy[1], size=number_events
        ),
        "observation_level_altitude": np.repeat(
            observation_level_altitude, number_events
        ),
        "core_position_on_observation_level_north": core_north,
        "core_position_on_observation_level_west": core_west,
        "azimuth": np.random.uniform(
            low=azimuth[0], high=azimuth[1], size=number_events
        ),
        "zenith_distance": sample_zenith_distance(
            min_zenith_distance=zenith_distance[0],
            max_zenith_distance=zenith_distance[1],
            size=number_events,
        ),
    }
