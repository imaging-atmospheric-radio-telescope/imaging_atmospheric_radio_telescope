from . import sites
from . import timing_and_sampling
from . import telescope
from . import telescopes
from . import signal
from . import lownoiseblock
from . import utils

import os
import rename_after_writing as rnw
import json_utils


def init(work_dir, site_key="karlsruhe", telescope_key="large_size_telescope"):
    """
    Initialize a working directory to simulate a specific telescope at a
    specific site.
    """
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
    """
    Load and compile the 'telescope' and the 'timing_and_sampling' from a
    working directory.
    """
    config = json_utils.tree.read(os.path.join(work_dir, "config"))
    config = utils.strip_dict(config, "comment")

    _lnb = lownoiseblock.init(key=config["telescope"]["lnb_key"])

    _mirror = telescope.make_mirror(**config["telescope"]["mirror"])
    _mirror_focal_ratio_1 = _mirror["focal_length_m"] / (
        2.0 * _mirror["outer_radius_m"]
    )
    print("f/D", _mirror_focal_ratio_1)
    _sensor = telescope.make_sensor(
        feed_horn_focal_ratio_1=_mirror_focal_ratio_1,
        low_noise_block_effective_area_m2=_lnb["effective_area_m2"],
        **config["telescope"]["sensor"],
    )

    _telescope = telescope.make_telescope(
        sensor=_sensor,
        mirror=_mirror,
        lnb=_lnb,
        speed_of_light_m_per_s=signal.SPEED_OF_LIGHT_M_PER_S,
    )

    _timing_and_sampling = timing_and_sampling.make_timing_from_lnb(
        lnb=_telescope["lnb"],
        **config["timing_and_sampling"],
    )

    return {
        "site": config["site"],
        "telescope": _telescope,
        "timing": _timing_and_sampling,
    }
