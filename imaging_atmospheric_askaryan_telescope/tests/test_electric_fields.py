import imaging_atmospheric_radio_telescope as iart
import numpy as np
import os
import tempfile


def test_raw_to_unified_and_back():
    for seed in range(10):
        E = iart.time_series.random(
            seed=seed, si_unit="V_per_m", num_components=3
        )
        raw = iart.electric_fields.to_coreas_electric_fields(E)
        back_E = iart.electric_fields.init_from_coreas_electric_fields(raw)
        iart.time_series.assert_almost_equal(actual=back_E, desired=E)
