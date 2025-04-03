"""
def simulate_mirror_electric_fields_calibration_source(
    out_dir,
    event_id,
    calibration_source,
    site,
    time_slice_duration_s,
    antenna_positions_obslvl_m,
    coreas_time_boundaries=corsika.coreas.DEFAULT_TIME_BOUNDARIES,
):
    ctb = coreas_time_boundaries
    antenna_path = os.path.join(out_dir, "electric_fields.tar")

    num_antennas = antenna_positions_obslvl_m.shape[0]

    exposure_time_s = ctb["time_upper_boundary_s"] - ctb["time_lower_boundary_s"]

    E_field = electric_fields.init(
        time_slice_duration_s=time_slice_duration_s,
        num_time_slices=int(np.ceil(exposure_time_s / time_slice_duration_s)),
        num_antennas=num_antennas,
        global_start_time_s=,
    )

    num_time_slices_source = int(
        np.ceil(2 * exposure_time_s / time_slice_duration_s)
    )
    _time, E_field_source = signal.make_sin(
        frequency=calibration_source["frequency_Hz"],
        time_slice_duration=time_slice_duration_s,
        num_time_slices=num_time_slices_source,
    )

    for i in range(num_antennas):
        antenna_position = antenna_positions_obslvl_m[i]
        distance = np.linalg.norm(antenna_position - calibration_source["position"])
        time_delay = distance / signal.SPEED_OF_LIGHT

        slice_delay = int(np.round(time_delay / time_slice_duration_s))

    electric_fields.write_tar(
        path=antenna_path, electric_fields=E_field,
    )
"""
