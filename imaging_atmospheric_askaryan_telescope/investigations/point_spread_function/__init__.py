from . import utils
from . import plot
from . import defocus
from . import stars
from . import power_image_analysis
from . import polarization_analysis

from ... import utils as iaat_utils
from ... import telescope
from ... import telescopes
from ... import sites
from ... import signal
from ... import production
from ... import time_series
from ... import electric_fields
from ... import lownoiseblock
from ... import timing_and_sampling


import numpy as np
import os
import json_utils
import rename_after_writing as rnw
import shutil
import glob


def init(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # telescopes
    telescopes_dir = os.path.join(config_dir, "telescopes")
    os.makedirs(telescopes_dir, exist_ok=True)

    for key in ["crome", "large_size_telescope"]:
        telescope_config = telescopes.init(key)
        with rnw.open(
            os.path.join(telescopes_dir, f"{key:s}.json"), "wt"
        ) as f:
            f.write(json_utils.dumps(telescope_config, indent=4))

    with rnw.open(os.path.join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(sites.init("karlsruhe"), indent=4))

    timing_config = {
        "oversampling": 6,
        "time_window_duration_s": 3.5e-08,
        "readout_sampling_rate_per_s": 250e6,
    }
    with rnw.open(
        os.path.join(config_dir, "timing_and_sampling.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(timing_config, indent=4))

    sc = {
        "telescopes": ["crome", "large_size_telescope"],
        "power_density_start_W_per_m2": 1e-12,
        "power_density_stop_W_per_m2": 3e-12,
        "scenarios": {},
    }
    sc["scenarios"]["representative_guide_stars"] = {
        "num": 5,
        "random_seed": 100,
    }
    sc["scenarios"]["central_feed_horn_scan"] = {"num": 8, "random_seed": 100}
    sc["scenarios"]["fully_inside_field_of_view"] = {
        "num": 8,
        "random_seed": 100,
    }
    sc["scenarios"]["on_edge_of_field_of_view"] = {
        "num": 8,
        "random_seed": 100,
    }
    sc["scenarios"]["outside_of_field_of_view"] = {
        "num": 8,
        "random_seed": 100,
    }

    with rnw.open(os.path.join(config_dir, "stars.json"), "wt") as f:
        f.write(json_utils.dumps(sc, indent=4))

    defocus_config = {
        "telescopes": ["large_size_telescope"],
        "start_object_distance_m": 4e3,
        "stop_object_distance_m": 20e3,
        "num": 16,
    }
    with rnw.open(os.path.join(config_dir, "defocus.json"), "wt") as f:
        f.write(json_utils.dumps(defocus_config, indent=4))


def run(work_dir, pool=None, logger=None):
    pool = utils.serial_pool_if_None(pool)
    logger = iaat_utils.stdout_logger_if_None(logger)
    config = utils.read_config(work_dir)

    logger.debug("make jobs for 'stars' ...")
    star_jobs = stars.make_jobs(work_dir=work_dir, config=config)
    star_jobs = stars.drop_finished_jobs(work_dir=work_dir, jobs=star_jobs)
    logger.debug(f"{len(star_jobs):d} jobs are missing and need to be run.")
    logger.debug("run jobs for 'stars' ...")
    pool.map(stars.run_job, star_jobs)

    logger.debug("make jobs for 'defocus' ...")
    defocus_jobs = defocus.make_jobs(work_dir=work_dir, config=config)
    defocus_jobs = defocus.drop_finished_jobs(
        work_dir=work_dir, jobs=star_jobs
    )
    logger.debug(f"{len(defocus_jobs):d} jobs are missing and need to be run.")
    logger.debug("run jobs for 'defocus' ...")
    pool.map(defocus.run_job, defocus_jobs)
