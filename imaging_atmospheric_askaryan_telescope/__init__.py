# Copyright 2017 Sebastian A. Mueller
from . import sites
from . import timing_and_sampling
from . import corsika
from . import telescope
from . import telescopes
from . import production
from . import signal
from . import electric_fields
from . import lownoiseblock
from . import theory
from . import utils

import os
import rename_after_writing as rnw
import json_utils


def init(work_dir, site_key="karlsruhe", telescope_key="large_size_telescope"):
    join = os.path.join

    config_dir = join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    with rnw.open(join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(sites.init(site_key), indent=4))

    with rnw.open(join(config_dir, "telescope.json"), "wt") as f:
        f.write(json_utils.dumps(telescopes.init(telescope_key), indent=4))

    with rnw.open(join(config_dir, "timing_and_sampling.json"), "wt") as f:
        f.write(
            json_utils.dumps(timing_and_sampling.default_config(), indent=4)
        )


def from_config(work_dir):

    config = json_utils.tree.read(os.path.join(work_dir, "config"))
    config = utils.strip_dict(config, "comment")

    _lnb = lownoiseblock.init(key=config["telescope"]["lnb_key"])
    _mirror = telescope.make_mirror(**config["telescope"]["mirror"])
    _sensor = telescope.make_sensor(**config["telescope"]["sensor"])

    _telescope = telescope.make_telescope(
        sensor=_sensor,
        mirror=_mirror,
        lnb=_lnb,
        speed_of_light_m_per_s=signal.SPEED_OF_LIGHT,
    )

    _timing_and_sampling = timing_and_sampling.make_timing_from_lnb(
        lnb=_telescope["lnb"],
        **config["timing_and_sampling"],
    )

    return {
        "telescope": _telescope,
        "timing_and_sampling": _timing_and_sampling,
    }


def simulate_event(work_dir, random_seed=1):

    res = from_config(work_dir=work_dir)
