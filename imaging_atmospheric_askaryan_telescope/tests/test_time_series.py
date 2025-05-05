import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import os
import tempfile


def test_write_read():
    with tempfile.TemporaryDirectory(prefix="askaryan_") as tmp:
        path = os.path.join(tmp, f"E.ts.tar")

        for seed in range(10):
            E = iaat.time_series.random(seed=seed)

            iaat.time_series.write(path=path, time_series=E)
            back_E = iaat.time_series.read(path=path)

            iaat.time_series.assert_almost_equal(actual=back_E, desired=E)
