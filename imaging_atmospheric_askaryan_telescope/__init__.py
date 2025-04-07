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


def init_telescope_and_timing(config):
    config = utils.strip_dict(config, "comment")

    lnb = lownoiseblock.init(key=config["lnb_key"])
    mir = telescope.make_mirror(**config["mirror"])
    sen = telescope.make_sensor(**config["sensor"])

    tel = telescope.make_telescope(
        sensor=sen,
        mirror=mir,
        lnb=lnb,
        speed_of_light_m_per_s=signal.SPEED_OF_LIGHT,
    )

    tim = timing_and_sampling.make_timing_from_lnb(
        lnb=tel["lnb"],
        **config["timing"],
    )

    return tel, tim


def init(work_dir, site_key="karlsruhe", telescope_key="large_size_telescope"):
    join = os.path.join

    config_dir = join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    with rnw.open(join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(sites.init(site_key), indent=4))

    with rnw.open(join(config_dir, "telescope.json"), "wt") as f:
        f.write(json_utils.dumps(telescopes.init(telescope_key), indent=4))