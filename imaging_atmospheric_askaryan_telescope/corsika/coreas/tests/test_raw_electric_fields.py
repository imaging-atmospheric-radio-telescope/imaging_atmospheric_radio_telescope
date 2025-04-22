import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import tempfile
import os


def test_write_read():
    with tempfile.TemporaryDirectory(prefix="askaryan_") as tmp:
        for seed in range(5):
            raw = iaat.corsika.coreas.raw_electric_fields.init_random(
                seed=seed
            )
            path = os.path.join(tmp, f"E_{seed:d}")
            iaat.corsika.coreas.raw_electric_fields.write(
                path=path, raw_electric_fields=raw
            )

            raw_back = iaat.corsika.coreas.raw_electric_fields.read(path=path)

            iaat.corsika.coreas.raw_electric_fields.assert_almost_eqaul(
                actual=raw_back, desired=raw
            )
