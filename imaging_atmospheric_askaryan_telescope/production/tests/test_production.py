import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np
import json_utils
import os
import tempfile


def test_simulate_mirror_electric_fields_manual():
    SPEED_OF_LIGHT = iaat.signal.SPEED_OF_LIGHT
    event_id = 103
    primary_particle = {
        "key": "gamma",
        "energy_GeV": 25.0,
        "azimuth_deg": 30,
        "zenith_distance_deg": 25,
        "core_north_m": 50.0,
        "core_west_m": 20.0,
    }
    time_slice_duration_s = 1e-9
    site = iaat.sites.init("karlsruhe")
    antenna_positions_obslvl_m = np.asarray([[0, 0, 0], [0, 1, 0]])
    coreas_time_boundaries = {
        "automatic_time_boundaries_s": 0,
        "time_lower_boundary_s": -100e3 / SPEED_OF_LIGHT,
        "time_upper_boundary_s": +100e3 / SPEED_OF_LIGHT,
    }

    with tempfile.TemporaryDirectory(prefix="askaryan_") as tmp:
        iaat.production.simulate_mirror_electric_fields_manual(
            out_dir=tmp,
            event_id=event_id,
            primary_particle=primary_particle,
            site=site,
            time_slice_duration_s=time_slice_duration_s,
            antenna_positions_obslvl_m=antenna_positions_obslvl_m,
            coreas_time_boundaries=coreas_time_boundaries,
        )
        electric_fields = iaat.electric_fields.read_tar(
            os.path.join(tmp, "electric_fields.tar")
        )

    np.testing.assert_almost_equal(
        actual=electric_fields["time_slice_duration_s"],
        desired=time_slice_duration_s,
    )
    assert electric_fields["num_antennas"] == 2

    output_duration_s = (
        coreas_time_boundaries["time_upper_boundary_s"]
        - coreas_time_boundaries["time_lower_boundary_s"]
    )

    assert (
        electric_fields["num_time_slices"]
        > 0.9 * output_duration_s / time_slice_duration_s
    )
    assert (
        electric_fields["num_time_slices"]
        < 1.1 * output_duration_s / time_slice_duration_s
    )

    np.testing.assert_almost_equal(
        actual=electric_fields["global_start_time_s"],
        desired=coreas_time_boundaries["time_lower_boundary_s"],
    )

    assert not np.any(np.isnan(electric_fields["electric_fields_V_per_m"]))
    assert np.any(electric_fields["electric_fields_V_per_m"] != 0)

    a0 = np.argmax(np.abs(electric_fields["electric_fields_V_per_m"][0, :]))
    a1 = np.argmax(np.abs(electric_fields["electric_fields_V_per_m"][1, :]))

    assert np.abs(a0 - a1) < (20e-9 / time_slice_duration_s)
