import imaging_atmospheric_askaryan_telescope as iaat
import numpy as np


def make_example_args():
    return {
        "azimuth_rad": 0,
        "zenith_rad": 0,
        "polarization_angle_rad": 0,
        "power_of_isotrop_and_point_like_emitter_W": 0,
        "distance_to_isotrop_and_point_like_emitter_m": 0,
        "emission_frequency_Hz": 0,
        "emission_duration_s": 0,
        "emission_ramp_up_duration_s": 0,
        "emission_ramp_down_duration_s": 0,
        "distance_to_define_time_zero": 0,
        "antenna_positions_asl_m": 0,
        "time_slice_duration_s": 0,
        "core_position_on_observation_level_north_m": 0,
        "core_position_on_observation_level_west_m": 0,
        "core_position_on_observation_level_asl_m": 0,
    }


def test_distance_between_plane_and_point():
    for z in np.linspace(-5, 6, 101):
        d = iaat.calibration_source.plane_wave_in_far_field.distance_between_plane_and_point(
            plane_support_vector=[0, 0, z],
            plane_normal_vector=[0, 0, 1],
            point=[0, 0, 0],
        )
        np.testing.assert_almost_equal(actual=d, desired=np.abs(z))


def test_geometry_setup_most_simple_case():
    geom = iaat.calibration_source.plane_wave_in_far_field.make_geometry_setup(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        polarization_angle_rad=0.0,
        distance_to_plane_defining_time_zero_m=1_000,
        core_position_vector_in_asl_frame_m=[0, 0, 0],
        antenna_position_vectors_in_asl_frame_m=[[0, 0, 0]],
    )

    np.testing.assert_array_almost_equal(
        actual=geom["plane_zero"]["support_vector_in_asl_frame"],
        desired=[0, 0, 1_000],
    )
    np.testing.assert_array_almost_equal(
        actual=geom["plane_zero"]["normal_vector_in_asl_frame"],
        desired=[0, 0, 1],
    )
    np.testing.assert_almost_equal(
        actual=geom["antenna_distances_to_plane_defining_time_zero_m"],
        desired=1_000,
    )


def test_power_setup():
    pows = iaat.calibration_source.plane_wave_in_far_field.make_power_setup(
        power_of_isotrop_and_point_like_emitter_W=3.0,
        distance_to_isotrop_and_point_like_emitter_m=100.0,
    )
    assert pows["pointing_vector_magnitude_W_per_m2"] == 3 / (
        4 * np.pi * 100**2
    )
    assert pows["electric_field_amplitue_V_per_m"] == np.sqrt(
        pows["pointing_vector_magnitude_W_per_m2"]
        * iaat.signal.VACUUM_IMPEDANCE
    )


def test_plane_wave():
    pwiff = iaat.calibration_source.plane_wave_in_far_field

    # setup
    # -----

    plane_wave_frequency_Hz = 9.75e9
    plane_wave_wavelength_m = (
        iaat.signal.SPEED_OF_LIGHT / plane_wave_frequency_Hz
    )
    oversampling = 6.0
    time_slice_duration_s = 1.0 / (oversampling * plane_wave_frequency_Hz)

    expected_spatial_resolution_with_given_oversampling_m = (
        iaat.signal.STANDARD_DEVIATION_OF_RECTANGULAR_FUNCTION
        * plane_wave_wavelength_m
        / oversampling
    )

    config = pwiff.make_config()
    config["sine_wave"]["emission_frequency_Hz"] = plane_wave_frequency_Hz
    config["geometry"]["antenna_position_vectors_in_asl_frame_m"] = [
        [0, 0, 0],
        [0, 0, 0.005],
        [0, 0, 0.010],
    ]

    geometry_setup = pwiff.make_geometry_setup(**config["geometry"])
    power_setup = pwiff.make_power_setup(**config["power"])

    # execution
    # ---------

    E = pwiff.plane_wave_in_far_field(
        geometry_setup=geometry_setup,
        power_setup=power_setup,
        sine_wave=config["sine_wave"],
        time_slice_duration_s=time_slice_duration_s,
    )

    # testing and asserting
    # ---------------------

    iaat.electric_fields.assert_valid(E)
    iaat.electric_fields.print_amplitudes(electric_fields=E)

    # test the phase of the sine wave
    # -------------------------------
    time_s = iaat.electric_fields.make_time_bin_centers(E)
    phase_shifts_rad = []
    for a in range(E["num_antennas"]):
        phase_shift_rad = iaat.signal.estimate_phase_angle_of_sine_wave(
            time_s=time_s,
            signal=np.linalg.norm(E["electric_fields_V_per_m"][a], axis=1),
            sine_wave_frequency_Hz=plane_wave_frequency_Hz,
        )
        phase_shifts_rad.append(phase_shift_rad)
    phase_shifts_rad = np.array(phase_shifts_rad)

    phase_shifts_1 = phase_shifts_rad / (2.0 * np.pi)
    phase_shifts_s = phase_shifts_1 / plane_wave_frequency_Hz
    phase_shifts_m = phase_shifts_s * iaat.signal.SPEED_OF_LIGHT

    for a in range(E["num_antennas"]):
        antenna_z_m = config["geometry"][
            "antenna_position_vectors_in_asl_frame_m"
        ][a][2]
        antenna_phase_shift_m = phase_shifts_m[a]

        delta_m = np.abs(antenna_z_m - antenna_phase_shift_m)
        assert delta_m <= expected_spatial_resolution_with_given_oversampling_m
