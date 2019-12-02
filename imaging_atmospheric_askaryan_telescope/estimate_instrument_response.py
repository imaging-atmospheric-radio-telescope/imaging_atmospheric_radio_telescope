import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import os
import random
import sun_grid_engine_map as sge


RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

CORSIKA_PATH = os.path.join(
    '/',
    'home',
    'hin',
    'relleums',
    'imaging_atmospheric_askaryan_telescope',
    'corsika_coreas_build',
    'corsika-77100',
    'run',
    'corsika77100Linux_QGSII_urqmd_coreas')

out_path = os.path.join(
    '/',
    'lfs',
    'l8',
    'hin',
    'relleums',
    'iaat_instrument_response')

imaging_reflector_config = {
    "antenna_areal_density": 4.0,
    "focal_length": 75,
    "aperture_radius": 25,
    "random_seed": 0}

OBJECT_DISTANCE = 10e3
image_sensor_distance = 1./(
    1./imaging_reflector_config['focal_length'] -
    1./OBJECT_DISTANCE)

image_sensor_config = {
    "pixel_inner_fov": np.deg2rad(0.11),
    "focal_length_of_imaging_system": 75,
    "fov": np.deg2rad(4.5),
    "image_sensor_distance": image_sensor_distance}

particles = {"gamma": 1, "proton": 14}

os.makedirs(out_path, exist_ok=True)
jobs = []
for particle in particles:
    particle_dir = os.path.join(out_path, particle)
    particle_id = particles[particle]
    run_config = {
        "primary_particle_id": particle_id,
        "number_events": 400,
        "energy": [5e2, 5e3],
        "zenith_distance": [0, np.deg2rad(1.5)],
        "azimuth": [0, np.deg2rad(360.)],
        "observation_level_altitude": 2200,
        "core_position_on_observation_level_max_scatter_radius": 150,
        "time_slice_duration": 2e-10}

    card = {
        "imaging_reflector": imaging_reflector_config.copy(),
        "image_sensor": image_sensor_config.copy(),
        "run": run_config.copy()}

    particle_jobs = iaat.map_and_reduce.make_jobs(
        corsika_coreas_path=CORSIKA_PATH,
        steering_card=card,
        out_dir=particle_dir)

    jobs += particle_jobs

random.shuffle(jobs)

sge.map(iaat.map_and_reduce.run_job, jobs)


