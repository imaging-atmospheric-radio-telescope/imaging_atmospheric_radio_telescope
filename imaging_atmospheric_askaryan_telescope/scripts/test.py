import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import os

F_LO = 10.7e9
F_SIM = 2 * F_LO

TIME_SLICE_DURATION = 1.0 / F_SIM

mirror = iaat.telescope.make_mirror(
    random_seed=0, focal_length=25.5, radius=8.5, probe_areal_density=0.1,
)

sensor = iaat.telescope.make_sensor(
    sensor_outer_radius=0.3, sensor_distance=25.5, feed_horn_inner_radius=0.03,
)

telescope = iaat.telescope.make_telescope(
    sensor=sensor, mirror=mirror, speed_of_light=3e8,
)

corsika_coreas_executable_path = os.path.join(
    "corsika_coreas_build",
    "corsika-77100",
    "run",
    "corsika77100Linux_QGSII_urqmd_coreas",
)

event_id = 124
event_path = "test{:06d}".format(event_id)

if not os.path.exists(event_path):
    iaat.run_corsika_coreas.simulate_event(
        corsika_coreas_executable_path=corsika_coreas_executable_path,
        out_dir=event_path,
        event_id=event_id,
        primary_particle_id=1,
        energy=1000,
        zenith_distance=0.0,
        azimuth=0.0,
        core_position_on_observation_level_north=10,
        core_position_on_observation_level_west=50,
        observation_level_altitude=2300,
        earth_magnetic_field_x_muT=12.5,
        earth_magnetic_field_z_muT=-25.9,
        time_slice_duration=TIME_SLICE_DURATION,
        telescope=telescope,
        num_time_slices=3000,
    )
