#!/usr/bin/env python
# Copyright 2017 Sebastian A. Mueller
"""
Make a video of the electric field components of all events in a run. The videos
are stored in the event directories.

Usage: make_videos --run_dir=DIR

Options:
    --run_dir=DIR       A run directory containing events.
"""
import docopt
import scoop
import os
import glob
from os.path import join
import matplotlib
matplotlib.use('Agg')
from askarian_telescope import plot
import askarian_telescope as at


def make_event_video(event_dir):
    try:
        event = at.telescope.Event(event_dir)
        image_slice_dir = join(event_dir, 'video_slices')
        plot.save_image_slices(event, image_slice_dir)
        plot.make_video_from_image_slices(
            image_slice_dir=image_slice_dir,
            out_path=join(event_dir, 'event_{:06d}'.format(event.id)),
            fps=12, 
            threads=1,
        )
    except Exception as e: 
        print('(Event ', event_dir, '):\n', e)
    return 1


def main():
    try:
        arguments = docopt.docopt(__doc__)
        run_dir = arguments['--run_dir']
        jobs = glob.glob(join(run_dir, 'event_*'))
        job_return_codes = list(scoop.futures.map(make_event_video, jobs))

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
