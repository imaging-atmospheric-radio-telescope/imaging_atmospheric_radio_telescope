import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot

import numpy as np
import os

F_LO = 10e9
F_SIM = 2 * F_LO

TIME_SLICE_DURATION = 1.0 / F_SIM

mirror = iaat.telescope.make_mirror(
    random_seed=0, focal_length=25.5, radius=8.5, probe_areal_density=5,
)

sensor = iaat.telescope.make_sensor(
    sensor_outer_radius=1.1, sensor_distance=25.5, feed_horn_inner_radius=0.03,
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

event_id = 126
event_path = "test{:06d}".format(event_id)

primary_particle = {
    "id": 1,
    "energy_GeV": 1000,
    "zenith_distance_rad": 0.0,
    "azimuth_rad": 0.0,
    "core_north_m": 10,
    "core_west_m": 50,
}


iaat.production.simulate_telescope_response(
    corsika_coreas_executable_path=corsika_coreas_executable_path,
    out_dir=event_path,
    event_id=event_id,
    primary_particle=primary_particle,
    site=iaat.sites.NAMIBIA,
    time_slice_duration=TIME_SLICE_DURATION,
    time_window_duration=400e-9,
    telescope=telescope,
    num_time_slices=4000,
)
"""
mirror_electric_fields = iaat.telescope.read_electric_fields(
    path=os.path.join(event_path, "mirror", "electric_fields")
)
iaat_plot.save_image_slices_electric_field(
    electric_fields=mirror_electric_fields,
    antenna_positions=telescope["mirror"]["antenna_positions"],
    path=os.path.join(event_path, "plot", "mirror"),
    time_slice_region_of_interest=np.arange(70, 700, 10),
    dpi=80,
    figsize=(12, 4),
)
"""
sensor_electric_fields = iaat.telescope.read_electric_fields(
    path=os.path.join(event_path, "sensor", "electric_fields")
)
iaat_plot.save_image_slices_electric_field(
    electric_fields=sensor_electric_fields,
    antenna_positions=telescope["sensor"]["antenna_positions"],
    path=os.path.join(event_path, "plot", "sensor"),
    time_slice_region_of_interest=np.arange(480, 580, 1),
    dpi=80,
    figsize=(12, 4),
)
