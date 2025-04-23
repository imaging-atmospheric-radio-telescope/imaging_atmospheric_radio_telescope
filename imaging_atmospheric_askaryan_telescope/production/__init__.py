# Copyright 2017 Sebastian A. Mueller
import numpy as np
import tempfile
import os
import json_utils
import rename_after_writing as rnw

from . import radio_from_airshower
from . import radio_from_plane_wave

from .. import telescope as simtelescope
from .. import electric_fields


def simulate_telescope_response(
    out_dir,
    source_config,
    site,
    telescope,
    timing,
):
    os.makedirs(out_dir, exist_ok=True)
    with rnw.open(os.path.join(out_dir, "source_config.json"), "wt") as f:
        f.write(json_utils.dumps(source_config, indent=4))

    # Electric fields on mirror
    # -------------------------
    if source_config["__type__"] == "airshower":
        radio_from_airshower.simulate_mirror_electric_fields(
            out_dir=out_dir,
            airshower_config=source_config,
            site=site,
            antenna_positions_obslvl_m=telescope["mirror"][
                "scatter_center_positions_m"
            ],
            timing=timing,
        )
    elif source_config["__type__"] == "plane_wave":
        radio_from_plane_wave.simulate_mirror_electric_fields(
            out_dir=out_dir,
            plane_wave_config=source_config,
            time_slice_duration_s=timing["electric_fields"][
                "time_slice_duration_s"
            ],
            antenna_positions_obslvl_m=telescope["mirror"][
                "scatter_center_positions_m"
            ],
            observation_level_asl_m=site["observation_level_asl_m"],
        )
    else:
        assert (
            False
        ), f"Source config __type__: {source_config['__type__']:s} is not known."

    # Electric fields in LNB feedhorns
    # --------------------------------
    sensor_dir = os.path.join(out_dir, "sensor")
    if not os.path.exists(sensor_dir):
        os.makedirs(sensor_dir)

        mirror_electric_fields = electric_fields.read_tar(
            path=os.path.join(out_dir, "mirror", "electric_fields.tar"),
        )

        sensor_electric_fields = (
            simtelescope.propagate_electric_field_from_mirror_to_sensor(
                telescope=telescope,
                mirror_electric_fields=mirror_electric_fields,
                num_time_slices=timing["electric_fields"]["sensor"][
                    "num_time_slices"
                ],
            )
        )

        electric_fields.write_tar(
            path=os.path.join(sensor_dir, "electric_fields.tar"),
            electric_fields=sensor_electric_fields,
        )

    # Electric response of LNBs electric fields
    # -----------------------------------------
    lnb_dir = os.path.join(out_dir, "lnb")
    if not os.path.exists(lnb_dir):
        pass
