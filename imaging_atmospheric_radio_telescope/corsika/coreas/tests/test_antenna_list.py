import imaging_atmospheric_radio_telescope as iart
import numpy as np


def assert_write_read_almost_equal(positions_asl_m):
    text = iart.corsika.coreas.antenna_list.dumps(
        positions_asl_m, prefix="hans"
    )
    positions_asl_m_back = iart.corsika.coreas.antenna_list.loads(text=text)
    np.testing.assert_array_almost_equal(
        actual=positions_asl_m_back,
        desired=positions_asl_m,
    )


def test_antenna_list_write_read():
    prng = prng = np.random.Generator(np.random.PCG64(4))

    for i in range(100):
        num_antennas = int(prng.uniform(low=0, high=100, size=1)[0])

        positions_asl_m = prng.uniform(size=num_antennas * 3).reshape(
            (num_antennas, 3)
        )

        assert_write_read_almost_equal(positions_asl_m=positions_asl_m)
