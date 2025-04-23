import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import os
import tempfile


def test_raw_to_unified_and_back():
    for seed in range(10):
        E = iaat.electric_fields.init_random(seed=seed)
        raw = iaat.electric_fields.to_coreas_electric_fields(E)
        back_E = iaat.electric_fields.init_from_coreas_electric_fields(raw)
        iaat.electric_fields.assert_almost_equal(actual=back_E, desired=E)


def test_write_read():

    FORMATS = {
        "tar": {
            "read": iaat.electric_fields.read_tar,
            "write": iaat.electric_fields.write_tar,
        },
        "loose_files": {
            "read": iaat.electric_fields.read,
            "write": iaat.electric_fields.write,
        },
    }

    with tempfile.TemporaryDirectory(prefix="askaryan_") as tmp:
        for form in FORMATS:

            tmp_E = os.path.join(tmp, f"E.{form:s}")
            for seed in range(10):
                E = iaat.electric_fields.init_random(seed=seed)

                FORMATS[form]["write"](path=tmp_E, electric_fields=E)
                back_E = FORMATS[form]["read"](path=tmp_E)

                iaat.electric_fields.assert_almost_equal(
                    actual=back_E, desired=E
                )
