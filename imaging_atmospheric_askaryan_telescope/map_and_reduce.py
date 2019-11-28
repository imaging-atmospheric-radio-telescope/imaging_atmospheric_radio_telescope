import os
import shutil
import imaging_atmospheric_askaryan_telescope as iaat


def run_job(job):
    try:
        os.makedirs(job['out_run_dir'], exist_ok=True)
        out_event_dir = os.path.join(
            job['out_run_dir'],
            'event_{:06d}'.format(job['event_id']))
        part_out_event_dir = out_event_dir + '.part'

        iaat.run_corsika_coreas.simulate_event(
            corsika_coreas_executable_path=job[
                'corsika_coreas_executable_path'],
            out_event_dir=part_out_event_dir,
            event_id=job['event_id'],
            primary_particle_id=job['primary_particle_id'],
            energy=job['energy'],
            zenith_distance=job['zenith_distance'],
            azimuth=job['azimuth'],
            observation_level_altitude=job['observation_level_altitude'],
            core_position_on_observation_level_north=job[
                'core_position_on_observation_level_north'],
            core_position_on_observation_level_west=job[
                'core_position_on_observation_level_west'],
            time_slice_duration=job['time_slice_duration'],
            imaging_reflector=job['imaging_reflector'],
            image_sensor=job['image_sensor'])

        shutil.move(part_out_event_dir, out_event_dir)
    except Exception as e:
        print('(Event ', job['event_id'], '):\n', e)

    return 1


def make_jobs(corsika_coreas_path, steering_card, out_dir):
    assert os.path.exists(corsika_coreas_path)
    sc = steering_card

    image_sensor = iaat.telescope.image_sensor_from_dict(sc['image_sensor'])
    imaging_reflector = iaat.telescope.imaging_reflector_from_dict(
        sc['imaging_reflector'])

    event_parameters = iaat.run_utils.draw_event_parameters(
        number_events=sc['run']['number_events'],
        primary_particle_id=sc['run']['primary_particle_id'],
        energy=sc['run']['energy'],
        azimuth=sc['run']['azimuth'],
        zenith_distance=sc['run']['zenith_distance'],
        observation_level_altitude=sc['run']['observation_level_altitude'],
        core_position_on_observation_level_max_scatter_radius=sc['run'][
            'core_position_on_observation_level_max_scatter_radius'],
        time_slice_duration=sc['run']['time_slice_duration'])

    jobs = []
    for idx in range(sc['run']['number_events']):
        job = {}
        for key in event_parameters:
            job[key] = event_parameters[key][idx]
        job['out_run_dir'] = out_dir
        job['corsika_coreas_executable_path'] = corsika_coreas_path
        job['image_sensor'] = image_sensor
        job['imaging_reflector'] = imaging_reflector
        jobs.append(job)
    return jobs
