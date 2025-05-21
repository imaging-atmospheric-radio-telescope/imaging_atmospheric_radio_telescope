import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np


def test_add_integet():
    first = np.zeros(5)
    first[0] = 1
    second = np.zeros(10)

    np.testing.assert_almost_equal(second[0], 0)
    iaat.signal.add_first_to_second_at_int(first=first, second=second, at=0)
    np.testing.assert_almost_equal(second[0], 1)

    np.testing.assert_array_almost_equal(second[1:], np.zeros(9))


def test_add_float_simple():
    first = np.zeros(5)
    first[0] = 1
    second = np.zeros(10)
    AT = 0.2

    np.testing.assert_almost_equal(second[0], 0)
    iaat.signal.add_first_to_second_at_float(first=first, second=second, at=AT)
    np.testing.assert_almost_equal(second[0], 1 - AT)
    np.testing.assert_almost_equal(second[1], AT)
    np.testing.assert_array_almost_equal(second[2:], np.zeros(8))


def test_add_float_negativ():
    for at in np.linspace(-0.9, -0.1, 13):
        first = np.zeros(5)
        first[0] = 1
        second = np.zeros(10)

        np.testing.assert_almost_equal(second[0], 0)
        iaat.signal.add_first_to_second_at_float(
            first=first, second=second, at=at
        )
        np.testing.assert_almost_equal(second[0], 1 - np.abs(at))
        np.testing.assert_array_almost_equal(second[1:], np.zeros(9))
