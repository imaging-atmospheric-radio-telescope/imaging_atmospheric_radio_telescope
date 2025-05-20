from . import utils
from . import plot
from . import defocus
from . import stars
from . import multis
from . import power_image_analysis
from . import polarization_analysis

from ... import utils as iaat_utils
from ... import logger as iaat_logger
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


def either(flag, x, y):
    if flag:
        return x
    else:
        return y


def resolve_mirror_oversampling(key):
    _map = {
        "low": 0.5,
        "mid": 1.0,
        "high": 2.0,
    }
    return _map[key]


def resolve_time_oversampling(key):
    _map = {
        "low": 3,
        "mid": 6,
        "high": 12,
    }
    return _map[key]


def resolve_feed_oversampling(key):
    _map = {
        "mid": 1,
        "high": 2,
    }
    return _map[key]


def init(
    work_dir,
    big=True,
    time_oversampling=6,
    mirror_oversampling=1,
    feed_horn_oversampling_order=1,
):
    assert 1.9 < time_oversampling
    assert 0.4 < mirror_oversampling

    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # telescopes
    telescopes_dir = os.path.join(config_dir, "telescopes")
    os.makedirs(telescopes_dir, exist_ok=True)

    TELESCOPE_KEYS = either(
        big,
        ["crome", "medium_size_telescope", "large_size_telescope"],
        ["crome", "medium_size_telescope"],
    )

    for key in TELESCOPE_KEYS:
        telescope_config = telescopes.init(key)
        telescope_config["mirror"][
            "scatter_center_areal_density_per_m2"
        ] *= mirror_oversampling
        telescope_config["sensor"][
            "feed_horn_oversampling_order"
        ] = feed_horn_oversampling_order
        with rnw.open(
            os.path.join(telescopes_dir, f"{key:s}.json"), "wt"
        ) as f:
            f.write(json_utils.dumps(telescope_config, indent=4))

    with rnw.open(os.path.join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(sites.init("karlsruhe"), indent=4))

    timing_config = {
        "oversampling": time_oversampling,
        "time_window_duration_s": 3.5e-08,
        "readout_sampling_rate_per_s": 250e6,
    }
    with rnw.open(
        os.path.join(config_dir, "timing_and_sampling.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(timing_config, indent=4))

    sc = {
        "telescopes": TELESCOPE_KEYS,
        "power_density_start_W_per_m2": 1e-12,
        "power_density_stop_W_per_m2": 3e-12,
        "scenarios": {},
    }
    sc["scenarios"]["representative_guide_stars"] = {
        "num": 5,
        "random_seed": 100,
    }
    sc["scenarios"]["central_feed_horn_scan"] = {
        "num": either(big, 80, 8),
        "random_seed": 100,
    }
    sc["scenarios"]["fully_inside_field_of_view"] = {
        "num": either(big, 400, 8),
        "random_seed": 100,
    }
    sc["scenarios"]["on_edge_of_field_of_view"] = {
        "num": either(big, 80, 4),
        "random_seed": 100,
    }
    sc["scenarios"]["fully_outside_field_of_view"] = {
        "num": either(big, 160, 8),
        "random_seed": 100,
    }

    with rnw.open(os.path.join(config_dir, "stars.json"), "wt") as f:
        f.write(json_utils.dumps(sc, indent=4))

    defocus_config = {
        "telescopes": either(
            big,
            ["medium_size_telescope", "large_size_telescope"],
            ["medium_size_telescope"],
        ),
        "start_sensor_distance_f": 0.99,
        "stop_sensor_distance_f": 1.05,
        "num": either(big, 64, 16),
    }
    with rnw.open(os.path.join(config_dir, "defocus.json"), "wt") as f:
        f.write(json_utils.dumps(defocus_config, indent=4))

    multis_config = {
        "telescopes": TELESCOPE_KEYS,
        "num_sources_per_event": 2,
        "random_seed": 120,
        "power_density_start_W_per_m2": 1e-12,
        "power_density_stop_W_per_m2": 3e-12,
        "num": either(big, 320, 16),
    }
    with rnw.open(os.path.join(config_dir, "multis.json"), "wt") as f:
        f.write(json_utils.dumps(multis_config, indent=4))


def run(work_dir, pool=None, logger=None):
    pool = utils.serial_pool_if_None(pool)
    logger = iaat_logger.LoggerStdout_if_logger_is_None(logger)
    config = utils.read_config(work_dir)

    logger.debug("make jobs for 'stars' ...")
    star_jobs = stars.make_jobs(work_dir=work_dir, config=config)
    logger.debug(f"{len(star_jobs):d} star jobs in total.")
    star_jobs = stars.drop_finished_jobs(work_dir=work_dir, jobs=star_jobs)
    logger.debug(f"{len(star_jobs):d} jobs are missing and need to be run.")
    logger.debug("run jobs for 'stars' ...")
    pool.map(stars.run_job, star_jobs)

    logger.debug("make jobs for 'defocus' ...")
    defocus_jobs = defocus.make_jobs(work_dir=work_dir, config=config)
    logger.debug(f"{len(defocus_jobs):d} defocus jobs in total.")
    defocus_jobs = defocus.drop_finished_jobs(
        work_dir=work_dir, jobs=defocus_jobs
    )
    logger.debug(f"{len(defocus_jobs):d} jobs are missing and need to be run.")
    logger.debug("run jobs for 'defocus' ...")
    pool.map(defocus.run_job, defocus_jobs)

    logger.debug("make jobs for 'multis' ...")
    multis_jobs = multis.make_jobs(work_dir=work_dir, config=config)
    logger.debug(f"{len(multis_jobs):d} multi jobs in total.")
    multis_jobs = multis.drop_finished_jobs(
        work_dir=work_dir, jobs=multis_jobs
    )
    logger.debug(f"{len(multis_jobs):d} jobs are missing and need to be run.")
    logger.debug("run jobs for 'multis' ...")
    pool.map(multis.run_job, multis_jobs)


def init_different_oversamplings(work_dir, combinations=None, big=True):
    if combinations is None:
        combinations = []
        combinations.append({"feed": "mid", "mirror": "mid", "time": "mid"})

        combinations.append({"feed": "mid", "mirror": "low", "time": "mid"})
        combinations.append({"feed": "mid", "mirror": "high", "time": "mid"})

        combinations.append({"feed": "mid", "mirror": "mid", "time": "low"})
        combinations.append({"feed": "mid", "mirror": "mid", "time": "high"})

        combinations.append({"feed": "high", "mirror": "mid", "time": "mid"})

        for combi in combinations:
            combi_dir = os.path.join(
                work_dir,
                f"mirror_{combi['mirror']:s}_feed_{combi['feed']:s}_time_{combi['time']:s}",
            )
            init(
                work_dir=combi_dir,
                big=big,
                time_oversampling=resolve_time_oversampling(combi["time"]),
                mirror_oversampling=resolve_mirror_oversampling(
                    combi["mirror"]
                ),
                feed_horn_oversampling_order=resolve_feed_oversampling(
                    combi["feed"]
                ),
            )


def run_different_oversamplings(work_dir, **kwargs):
    work_dirs = glob.glob(os.path.join(work_dir, "mirror_*_feed_*_time_*"))

    for path in work_dirs:
        run(work_dir=path, **kwargs)
