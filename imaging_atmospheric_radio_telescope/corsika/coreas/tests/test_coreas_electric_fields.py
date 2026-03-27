import imaging_atmospheric_radio_telescope as iart
import numpy as np
import tempfile
import os


def test_write_read():
    with tempfile.TemporaryDirectory(prefix="iart_") as tmp:
        for seed in range(5):
            raw = iart.corsika.coreas.coreas_electric_fields.init_random(
                seed=seed
            )
            path = os.path.join(tmp, f"E_{seed:d}")
            iart.corsika.coreas.coreas_electric_fields.write(
                path=path, coreas_electric_fields=raw
            )

            raw_back = iart.corsika.coreas.coreas_electric_fields.read(
                path=path
            )

            iart.corsika.coreas.coreas_electric_fields.assert_almost_eqaul(
                actual=raw_back, desired=raw
            )
