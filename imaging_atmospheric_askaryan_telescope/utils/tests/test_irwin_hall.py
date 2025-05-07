import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np


def test_std_and_mean():
    prng = np.random.Generator(np.random.PCG64(101))

    for size in [10, 100, 1000]:
        for mean in [-1.0, 0.0, 1337.0]:
            for std in [0.0, 1e-6, 15.0]:

                x = iaat.utils.normal_approximation(
                    prng=prng,
                    mean=mean,
                    std=std,
                    size=size,
                )

                assert x.shape[0] == size
                assert np.abs(np.mean(x) - mean) <= 3 * std / np.sqrt(size)
                assert np.std(x) - std <= 3 * std / np.sqrt(size)


def test_order():
    prng = np.random.Generator(np.random.PCG64(101))

    for order in [3, 6, 12]:
        x = iaat.utils.normal_approximation(
            prng=prng,
            mean=0.0,
            std=1.0,
            size=1_000_000,
            irwin_hall_order=order,
        )

        amax = np.max(np.abs(x))
        assert amax < order / 2
