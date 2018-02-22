#!/usr/bin/env python
# Copyright 2017 Sebastian A. Mueller
"""
Usage: run_simulation --out_dir=DIR [--steering_card=PATH] [--corsika_coreas=PATH]

Options:
    --out_dir=DIR
    --steering_card=PATH        [default: ./example_steering_card.json]
    --corsika_coreas=PATH       [default: ../corsika_coreas_build/corsika-75600/run/corsika75600Linux_QGSII_urqmd_coreas]
"""
import docopt
import scoop
import os
from os.path import join
import shutil
import json
from imaging_atmospheric_askaryan_telescope import run_corsika_coreas as rcc
from imaging_atmospheric_askaryan_telescope import telescope


def make_event(job):
    try:
        os.makedirs(job['out_run_dir'], exist_ok=True)
        out_event_dir = join(
            job['out_run_dir'], 'event_{:06d}'.format(job['event_id'])
        )
        part_out_event_dir = out_event_dir + '.part'

        rcc.simulate_event(
            corsika_coreas_executable_path=job['corsika_coreas_executable_path'],
            out_event_dir=part_out_event_dir,
            event_id=job['event_id'],
            primary_particle_id=job['primary_particle_id'],
            energy=job['energy'],
            zenith_distance=job['zenith_distance'],
            azimuth=job['azimuth'],
            observation_level_altitude=job['observation_level_altitude'],
            core_position_on_observation_level_north=job['core_position_on_observation_level_north'],
            core_position_on_observation_level_west=job['core_position_on_observation_level_west'],
            time_slice_duration=job['time_slice_duration'],
            imaging_reflector=job['imaging_reflector'],
            image_sensor=job['image_sensor'],
        )

        shutil.move(part_out_event_dir, out_event_dir)
    except Exception as e:
        print('(Event ', job['event_id'], '):\n', e)

    return 1


def main():
    try:
        arguments = docopt.docopt(__doc__)
        steering_card_path = arguments['--steering_card']
        out_run_dir = arguments['--out_dir']
        corsika_coreas_executable_path = arguments['--corsika_coreas']
        assert os.path.exists(corsika_coreas_executable_path)
        with open(steering_card_path, 'rt') as fin:
            steering_card = json.loads(fin.read())

        image_sensor = telescope.image_sensor_from_dict(
            steering_card['image_sensor']
        )
        imaging_reflector = telescope.imaging_reflector_from_dict(
            steering_card['imaging_reflector']
        )

        sc = steering_card
        event_parameters = rcc.event_parameter_distribution(
            number_events=sc['run']['number_events'],
            primary_particle_id=sc['run']['primary_particle_id'],
            energy=sc['run']['energy'],
            azimuth=sc['run']['azimuth'],
            zenith_distance=sc['run']['zenith_distance'],
            observation_level_altitude=sc['run']['observation_level_altitude'],
            core_position_on_observation_level_max_scatter_radius=sc['run']['core_position_on_observation_level_max_scatter_radius'],
            time_slice_duration=sc['run']['time_slice_duration'],
        )

        jobs = []
        for idx in range(sc['run']['number_events']):
            job = {}
            for key in event_parameters:
                job[key] = event_parameters[key][idx]
            job['out_run_dir'] = out_run_dir
            job['corsika_coreas_executable_path'] = corsika_coreas_executable_path
            job['image_sensor'] = image_sensor
            job['imaging_reflector'] = imaging_reflector
            jobs.append(job)

        os.makedirs(out_run_dir)
        shutil.copy(steering_card_path, join(out_run_dir, 'steering_card.json'))

        list(scoop.futures.map(make_event, jobs))

    except docopt.DocoptExit as e:
        print(e)


if __name__ == '__main__':
    main()
