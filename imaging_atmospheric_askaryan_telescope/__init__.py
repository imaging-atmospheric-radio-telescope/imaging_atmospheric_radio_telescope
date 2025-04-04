# Copyright 2017 Sebastian A. Mueller
from . import sites
from . import timing_and_sampling
from . import corsika
from . import telescope
from . import production
from . import signal
from . import electric_fields
from . import lownoiseblock
from . import theory
from . import utils


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
