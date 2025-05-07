import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np


def test_make_sine():
    PI = np.pi

    for phase in [
        0 * PI,
        (1 / 2) * PI,
        (2 / 2) * PI,
        (3 / 2) * PI,
        (4 / 2) * PI,
    ]:
        t, a = iaat.signal.make_sin(
            frequency=1.0,
            time_slice_duration=1e-3,
            num_time_slices=10_000,
            phase=phase,
        )
        np.testing.assert_almost_equal(actual=t[0], desired=0)
        np.testing.assert_almost_equal(actual=a[0], desired=np.sin(phase))
        assert np.all(a <= 1.0)
        assert np.all(a >= -1.0)
