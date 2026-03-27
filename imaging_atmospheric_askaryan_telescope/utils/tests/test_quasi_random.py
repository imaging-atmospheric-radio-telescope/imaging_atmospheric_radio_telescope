import imaging_atmospheric_radio_telescope as iart
import numpy as np


def test_same_seed_yields_same_results():
    qrng_a = iart.utils.QuasiRandomGenerator(seed=1)

    A = []
    for i in range(1000):
        a = qrng_a.uniform()
        A.append(a)

    qrng_b = iart.utils.QuasiRandomGenerator(seed=1)

    B = []
    for i in range(1000):
        b = qrng_b.uniform()
        B.append(b)

    np.testing.assert_array_almost_equal(actual=A, desired=B)


def test_shape():
    qrng = iart.utils.QuasiRandomGenerator(seed=1)

    a = qrng.uniform(size=None)
    assert np.array(a).shape == ()

    a = qrng.uniform(size=0)
    assert np.array(a).shape == (0,)

    a = qrng.uniform(size=1)
    assert np.array(a).shape == (1,)

    a = qrng.uniform(size=2)
    assert np.array(a).shape == (2,)


def test_low_high():
    qrng = iart.utils.QuasiRandomGenerator(seed=1)

    for low in np.linspace(-1, 1, 10):
        for high in low + np.linspace(0, 2, 10):
            assert high >= low
            a = qrng.uniform(low=low, high=high, size=1024)

            assert np.all(a >= low)
            assert np.all(a <= high)
