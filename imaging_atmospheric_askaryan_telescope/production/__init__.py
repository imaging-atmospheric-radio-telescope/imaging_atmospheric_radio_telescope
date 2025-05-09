# Copyright 2017 Sebastian A. Mueller
import numpy as np
import tempfile
import os
import json_utils
import rename_after_writing as rnw

from . import radio_from_airshower
from . import radio_from_plane_wave

from .. import telescope as simtelescope
from .. import electric_fields
from .. import signal
from .. import time_series
from .. import lownoiseblock
from .. import utils


def simulate_telescope_response(
    out_dir,
    source_config,
    site,
    telescope,
    timing,
    thermal_noise_random_seed,
    readout_random_seed,
    camera_lnb_random_seed,
    stop_after_section=None,
    logger=None,
):
    logger = utils.stdout_logger_if_None(logger)
    os.makedirs(out_dir, exist_ok=True)
    with rnw.open(os.path.join(out_dir, "source_config.json"), "wt") as f:
        f.write(json_utils.dumps(source_config, indent=4))

    # Electric fields on mirror
    # -------------------------
    mirror_dir = os.path.join(out_dir, "mirror")
    if not os.path.exists(mirror_dir):
        if source_config["__type__"] == "airshower":
            with rnw.Directory(
                mirror_dir
            ) as tmp_mirror_dir, utils.LoggerStartStop(
                logger, "Simulating air shower using CORSIKA CoREAS"
            ) as _:
                radio_from_airshower.assert_config_is_valid(source_config)
                radio_from_airshower.simulate_mirror_electric_fields(
                    mirror_dir=tmp_mirror_dir,
                    airshower_config=source_config,
                    site=site,
                    antenna_positions_obslvl_m=telescope["mirror"][
                        "scatter_center_positions_m"
                    ],
                    timing=timing,
                )

        elif source_config["__type__"] == "plane_wave":
            with rnw.Directory(
                mirror_dir
            ) as tmp_mirror_dir, utils.LoggerStartStop(
                logger, "Simulating plane wave from calibration source"
            ) as _:
                radio_from_plane_wave.simulate_mirror_electric_fields(
                    mirror_dir=tmp_mirror_dir,
                    plane_waves=source_config["plane_waves"],
                    time_slice_duration_s=timing["electric_fields"][
                        "time_slice_duration_s"
                    ],
                    antenna_positions_obslvl_m=telescope["mirror"][
                        "scatter_center_positions_m"
                    ],
                    observation_level_asl_m=site["observation_level_asl_m"],
                )

        else:
            assert (
                False
            ), f"Source config __type__: {source_config['__type__']:s} is not known."

    if stop_after_section == "mirror":
        return

    # Electric fields entering feed horns
    # -----------------------------------
    feed_horns_dir = os.path.join(out_dir, "feed_horns")
    if not os.path.exists(feed_horns_dir):
        with rnw.Directory(feed_horns_dir) as tmp_dir, utils.LoggerStartStop(
            logger, "Propagating electric fields from mirror to feed horns"
        ) as _:
            E_mirror = time_series.read(
                path=os.path.join(out_dir, "mirror", "electric_fields.tar"),
            )
            E_sensor = (
                simtelescope.propagate_electric_field_from_mirror_to_sensor(
                    telescope=telescope,
                    mirror_electric_fields=E_mirror,
                    num_time_slices=timing["electric_fields"]["sensor"][
                        "num_time_slices"
                    ],
                )
            )
            time_series.write(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                time_series=E_sensor,
            )

    if stop_after_section == "feed_horns":
        return

    # Electric fields entering lnbs
    # -----------------------------
    lnb_input_dir = os.path.join(out_dir, "lnb_input")
    if not os.path.exists(lnb_input_dir):
        with rnw.Directory(lnb_input_dir) as tmp_dir, utils.LoggerStartStop(
            logger, "Propagating electric fields through feed horns"
        ) as _:
            E_sensor = time_series.read(
                path=os.path.join(
                    out_dir, "feed_horns", "electric_fields.tar"
                ),
            )
            E_lnb_input = simulate_electric_field_leaving_feed_horns(
                electric_fields_entering_feed_horns=E_sensor,
                feed_horn_area_m2=telescope["sensor"]["feed_horn_area_m2"],
                lnb_effective_area_m2=telescope["lnb"]["effective_area_m2"],
                feed_horn_transmission=telescope["sensor"][
                    "feed_horn_transmission"
                ],
            )
            time_series.write(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                time_series=E_lnb_input,
            )

    if stop_after_section == "lnb_input":
        return

    # Signal electric fields leaving lnbs
    # -----------------------------------
    lnb_signal_output_dir = os.path.join(out_dir, "lnb_signal_output")
    if not os.path.exists(lnb_signal_output_dir):
        with rnw.Directory(
            lnb_signal_output_dir
        ) as tmp_dir, utils.LoggerStartStop(
            logger, "Simulating signal leaving low noise block converters"
        ) as _:
            E_lnb_input = time_series.read(
                path=os.path.join(out_dir, "lnb_input", "electric_fields.tar"),
            )
            E_lnb_signal_output = lownoiseblock.simulate_mixer(
                lnb_input_electric_fields=E_lnb_input,
                local_oscillator_frequency_Hz=telescope["lnb"][
                    "local_oscillator_frequency_Hz"
                ],
                local_oscillator_frequency_std_Hz=telescope["lnb"][
                    "local_oscillator_frequency_std_Hz"
                ],
                intermediate_frequency_start_Hz=telescope["lnb"][
                    "intermediate_frequency_start_Hz"
                ],
                intermediate_frequency_stop_Hz=telescope["lnb"][
                    "intermediate_frequency_stop_Hz"
                ],
                random_seed=camera_lnb_random_seed,
            )
            time_series.write(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                time_series=E_lnb_signal_output,
            )

    if stop_after_section == "lnb_signal_output":
        return

    # Noise electric fields leaving lnbs
    # ----------------------------------
    lnb_noise_output_dir = os.path.join(out_dir, "lnb_noise_output")
    if not os.path.exists(lnb_noise_output_dir):
        with rnw.Directory(
            lnb_noise_output_dir
        ) as tmp_dir, utils.LoggerStartStop(
            logger, "Simulating noise leaving low noise block converters"
        ) as _:
            prng = np.random.Generator(
                np.random.PCG64(thermal_noise_random_seed)
            )

            E_lnb_signal = time_series.read(
                path=os.path.join(out_dir, "lnb_input", "electric_fields.tar"),
            )
            signal_duration_s = E_lnb_signal.exposure_duration_s

            E_lnb_noise = (
                time_series.zeros_like_other_but_with_overhead_in_time(
                    other=E_lnb_signal,
                    leading_overhead_duration_s=signal_duration_s / 2,
                    trailing_overhead_duration_s=signal_duration_s / 2,
                )
            )

            electric_field_thermal_noise_amplitude_V_per_m = signal.calculate_electric_field_strength_of_thermal_noise_V_per_m(
                antenna_temperature_K=telescope["lnb"]["noise_temperature_K"],
                antenna_bandwidth_Hz=telescope["lnb"][
                    "intermediate_bandwidth_Hz"
                ],
                antenna_effective_area_m2=telescope["lnb"][
                    "effective_area_m2"
                ],
            )
            for channel in range(E_lnb_noise.num_channels):
                E_lnb_noise[channel] = prng.normal(
                    loc=0.0,
                    scale=electric_field_thermal_noise_amplitude_V_per_m,
                    size=(
                        E_lnb_noise.num_time_slices,
                        E_lnb_noise.num_components,
                    ),
                )

            for channel in range(E_lnb_noise.num_channels):
                lnb_noise_output_power_W = signal.calculate_antenna_power_W(
                    effective_area_m2=telescope["lnb"]["effective_area_m2"],
                    electric_field_V_per_m=E_lnb_noise[channel],
                )
                assert (
                    0.9
                    < (
                        telescope["lnb"]["noise_power_W"]
                        / np.mean(lnb_noise_output_power_W)
                    )
                    < 1.1
                )

            assert E_lnb_noise.dtype == E_lnb_signal.dtype
            time_series.write(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                time_series=E_lnb_noise,
            )

    if stop_after_section == "lnb_noise_output":
        return

    # Total electric fields leaving lnbs
    # ----------------------------------
    lnb_signal_and_noise_output_dir = os.path.join(
        out_dir, "lnb_signal_and_noise_output"
    )
    if not os.path.exists(lnb_signal_and_noise_output_dir):
        with rnw.Directory(
            lnb_signal_and_noise_output_dir
        ) as tmp_dir, utils.LoggerStartStop(
            logger,
            "Adding noise and signal leaving low noise block converters",
        ) as _:
            E_lnb_signal = time_series.read(
                path=os.path.join(
                    out_dir, "lnb_signal_output", "electric_fields.tar"
                )
            )

            E_lnb_noise = time_series.read(
                path=os.path.join(
                    out_dir, "lnb_noise_output", "electric_fields.tar"
                )
            )

            E_lnb_noise_and_signal = E_lnb_noise.add(E_lnb_signal)

            time_series.write(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                time_series=E_lnb_noise_and_signal,
            )

    if stop_after_section == "lnb_signal_and_noise_output":
        return

    # Lnb Readout
    # -----------
    lnb_readout_dir = os.path.join(out_dir, "lnb_readout")
    if not os.path.exists(lnb_readout_dir):
        with rnw.Directory(lnb_readout_dir) as tmp_dir, utils.LoggerStartStop(
            logger, "Simulating Readout of LNBs"
        ):
            E_lnb_output = time_series.read(
                path=os.path.join(
                    out_dir,
                    "lnb_signal_and_noise_output",
                    "electric_fields.tar",
                )
            )
            Ene_readout = simulate_readout(
                electric_fields_leaving_lnbs=E_lnb_output,
                telescope=telescope,
                timing=timing,
                random_seed=readout_random_seed,
            )
            time_series.write(
                path=os.path.join(tmp_dir, "energies.ts.tar"),
                time_series=Ene_readout,
            )

    if stop_after_section == "lnb_readout":
        return


def simulate_electric_field_leaving_feed_horns(
    electric_fields_entering_feed_horns,
    feed_horn_area_m2,
    lnb_effective_area_m2,
    feed_horn_transmission,
):
    assert feed_horn_area_m2 > 0
    assert lnb_effective_area_m2 > 0
    assert feed_horn_transmission >= 0

    feed_horn_geometric_gain = feed_horn_area_m2 / lnb_effective_area_m2
    feed_horn_gain = feed_horn_geometric_gain * feed_horn_transmission

    electric_field_leaving_feed_horns = time_series.zeros_like(
        other=electric_fields_entering_feed_horns
    )

    electric_field_leaving_feed_horns[:] = (
        np.sqrt(feed_horn_gain).astype(np.float32)
        * electric_fields_entering_feed_horns[:]
    )
    return electric_field_leaving_feed_horns


def simulate_readout(
    electric_fields_leaving_lnbs,
    telescope,
    timing,
    random_seed,
):
    E_lnb = electric_fields_leaving_lnbs
    prng = np.random.Generator(np.random.PCG64(random_seed))

    # E field to power
    total_power_leaving_lnb = signal.calculate_antenna_power_W(
        effective_area_m2=telescope["lnb"]["effective_area_m2"],
        electric_field_V_per_m=E_lnb[:],
    )

    # integrate power_leaving_lnb over time for readout
    # -------------------------------------------------
    total_power_sliding_integral = np.zeros(
        shape=total_power_leaving_lnb.shape
    )

    numT = timing["readout"]["integrates_num_simulation_time_slices"]
    simulation_time_slice_duration = timing["electric_fields"][
        "time_slice_duration_s"
    ]

    for t in range(E_lnb.num_time_slices - numT):
        w = np.sum(total_power_leaving_lnb[:, t : t + numT, :], axis=1)
        total_power_sliding_integral[:, t, :] = (
            w * simulation_time_slice_duration
        )

    simulation_time_slices_which_are_sampled_by_readout = np.arange(
        0,
        E_lnb.num_time_slices,
        numT,
    )
    random_offset_of_readout_wrt_global_time_num_time_slices = int(
        prng.uniform(low=0, high=numT)
    )
    num_readout_frames = (
        len(simulation_time_slices_which_are_sampled_by_readout) - 1
    )

    readout_time_slice_duration_s = (
        timing["readout"]["integrates_num_simulation_time_slices"]
        * timing["electric_fields"]["time_slice_duration_s"]
    )

    readout_global_start_time_s = (
        random_offset_of_readout_wrt_global_time_num_time_slices
        * timing["electric_fields"]["time_slice_duration_s"]
        + E_lnb.global_start_time_s
    )

    readout_energy = time_series.zeros(
        time_slice_duration_s=readout_time_slice_duration_s,
        num_time_slices=num_readout_frames,
        num_channels=telescope["sensor"]["num_feed_horns"],
        num_components=2,
        global_start_time_s=readout_global_start_time_s,
        si_unit="J",
    )

    for i in range(num_readout_frames):
        simulation_time_slice = (
            simulation_time_slices_which_are_sampled_by_readout[i]
        )
        simulation_time_slice += (
            random_offset_of_readout_wrt_global_time_num_time_slices
        )
        x_comp_energy = total_power_sliding_integral[
            :, simulation_time_slice, 0
        ]
        y_comp_energy = total_power_sliding_integral[
            :, simulation_time_slice, 1
        ]
        readout_energy[:, i, 0] = x_comp_energy
        readout_energy[:, i, 1] = y_comp_energy

    return readout_energy
