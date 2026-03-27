import imaging_atmospheric_radio_telescope as iart
import numpy as np
import os
import tempfile


def test_write_read():
    with tempfile.TemporaryDirectory(prefix="iart_") as tmp:
        path = os.path.join(tmp, f"E.ts.tar")

        for seed in range(10):
            E = iart.time_series.random(seed=seed)

            iart.time_series.write(path=path, time_series=E)
            back_E = iart.time_series.read(path=path)

            iart.time_series.assert_almost_equal(actual=back_E, desired=E)
