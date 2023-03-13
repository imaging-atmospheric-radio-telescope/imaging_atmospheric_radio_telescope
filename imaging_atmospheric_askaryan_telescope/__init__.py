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


def init_telescope_and_timing(config):
    config = _strip_dict(config, "comment")

    lnb = lownoiseblock.init(lnb_name=config["lnb_name"])
    mir = telescope.make_mirror(**config["mirror"])
    sen = telescope.make_sensor(**config["sensor"])

    tel = telescope.make_telescope(
        sensor=sen,
        mirror=mir,
        lnb=lnb,
        speed_of_light_m_per_s=signal.SPEED_OF_LIGHT,
    )
    tel["transmission_from_air_into_feed_horn"] = config[
        "transmission_from_air_into_feed_horn"
    ]

    tim = timing_and_sampling.make_timing_from_lnb(
        lnb=tel["lnb"], **config["timing"],
    )

    return tel, tim


def _strip_dict(obj, strip):
    out = {}
    for key in obj:
        if key != strip:
            item = obj[key]
            if isinstance(item, dict):
                out[key] = _strip_dict(obj=item, strip=strip)
            else:
                out[key] = item
    return out
