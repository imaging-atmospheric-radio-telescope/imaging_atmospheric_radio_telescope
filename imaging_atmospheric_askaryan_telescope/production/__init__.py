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


def simulate_telescope_response(
    out_dir,
    source_config,
    site,
    telescope,
    timing,
    thermal_noise_random_seed,
):
    os.makedirs(out_dir, exist_ok=True)
    with rnw.open(os.path.join(out_dir, "source_config.json"), "wt") as f:
        f.write(json_utils.dumps(source_config, indent=4))

    # Electric fields on mirror
    # -------------------------
    mirror_dir = os.path.join(out_dir, "mirror")
    if not os.path.exists(mirror_dir):
        if source_config["__type__"] == "airshower":
            print(
                "Simulating air shower using CORSIKA CoREAS ... ",
                end="",
                flush=True,
            )
            radio_from_airshower.assert_config_is_valid(source_config)
            radio_from_airshower.simulate_mirror_electric_fields(
                out_dir=out_dir,
                airshower_config=source_config,
                site=site,
                antenna_positions_obslvl_m=telescope["mirror"][
                    "scatter_center_positions_m"
                ],
                timing=timing,
            )
            print("Done.")

        elif source_config["__type__"] == "plane_wave":
            print(
                "Simulating plane wave from calibration source ... ",
                end="",
                flush=True,
            )
            radio_from_plane_wave.simulate_mirror_electric_fields(
                out_dir=out_dir,
                plane_wave_config=source_config,
                time_slice_duration_s=timing["electric_fields"][
                    "time_slice_duration_s"
                ],
                antenna_positions_obslvl_m=telescope["mirror"][
                    "scatter_center_positions_m"
                ],
                observation_level_asl_m=site["observation_level_asl_m"],
            )
            print("Done.")

        else:
            assert (
                False
            ), f"Source config __type__: {source_config['__type__']:s} is not known."

    # Electric fields entering feed horns
    # -----------------------------------
    feed_horns_dir = os.path.join(out_dir, "feed_horns")
    if not os.path.exists(feed_horns_dir):
        with rnw.Directory(feed_horns_dir) as tmp_dir:
            print(
                "Propagating electric fields from mirror to feed horns ... ",
                end="",
                flush=True,
            )
            mirror_electric_fields = electric_fields.read_tar(
                path=os.path.join(out_dir, "mirror", "electric_fields.tar"),
            )
            sensor_electric_fields = (
                simtelescope.propagate_electric_field_from_mirror_to_sensor(
                    telescope=telescope,
                    mirror_electric_fields=mirror_electric_fields,
                    num_time_slices=timing["electric_fields"]["sensor"][
                        "num_time_slices"
                    ],
                )
            )
            electric_fields.write_tar(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                electric_fields=sensor_electric_fields,
            )
            print("Done.")

    # Electric fields entering lnbs
    # -----------------------------
    lnb_input_dir = os.path.join(out_dir, "lnb_input")
    if not os.path.exists(lnb_input_dir):
        with rnw.Directory(lnb_input_dir) as tmp_dir:
            print(
                "Propagating electric fields through feed horns ... ",
                end="",
                flush=True,
            )
            sensor_electric_fields = electric_fields.read_tar(
                path=os.path.join(
                    out_dir, "feed_horns", "electric_fields.tar"
                ),
            )
            lnb_input_electric_fields = (
                simulate_electric_field_leaving_feed_horns(
                    electric_fields_entering_feed_horns=sensor_electric_fields,
                    feed_horn_area_m2=telescope["sensor"]["feed_horn_area_m2"],
                    lnb_effective_area_m2=telescope["lnb"][
                        "effective_area_m2"
                    ],
                    feed_horn_transmission=telescope["sensor"][
                        "feed_horn_transmission"
                    ],
                )
            )
            electric_fields.write_tar(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                electric_fields=lnb_input_electric_fields,
            )
            print("Done.")

    # Signal electric fields leaving lnbs
    # -----------------------------------
    lnb_signal_output_dir = os.path.join(out_dir, "lnb_signal_output")
    if not os.path.exists(lnb_signal_output_dir):
        with rnw.Directory(lnb_signal_output_dir) as tmp_dir:
            print(
                "Simulating signal leaving low noise block converters ... ",
                end="",
                flush=True,
            )
            lnb_input_electric_fields = electric_fields.read_tar(
                path=os.path.join(out_dir, "lnb_input", "electric_fields.tar"),
            )
            lnb_signal_output_electric_fields = (
                electric_fields.init_zeros_like(
                    other=lnb_input_electric_fields
                )
            )
            lnb_signal_output_electric_fields["electric_fields_V_per_m"] = (
                signal.lnb_mixer(
                    amplitudes=lnb_input_electric_fields[
                        "electric_fields_V_per_m"
                    ],
                    time_slice_duration=timing["electric_fields"][
                        "time_slice_duration_s"
                    ],
                    local_oscillator_frequency=telescope["lnb"][
                        "local_oscillator_frequency_Hz"
                    ],
                    intermediate_frequency_start=telescope["lnb"][
                        "intermediate_frequency_start_Hz"
                    ],
                    intermediate_frequency_stop=telescope["lnb"][
                        "intermediate_frequency_stop_Hz"
                    ],
                )
            )
            lnb_signal_output_electric_fields["electric_fields_V_per_m"] = (
                lnb_signal_output_electric_fields[
                    "electric_fields_V_per_m"
                ].astype(np.float32)
            )
            electric_fields.write_tar(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                electric_fields=lnb_signal_output_electric_fields,
            )
            print("Done.")

    # Noise electric fields leaving lnbs
    # ----------------------------------
    lnb_noise_output_dir = os.path.join(out_dir, "lnb_noise_output")
    if not os.path.exists(lnb_noise_output_dir):
        with rnw.Directory(lnb_noise_output_dir) as tmp_dir:
            print(
                "Simulating noise leaving low noise block converters ... ",
                end="",
                flush=True,
            )
            prng = np.random.Generator(
                np.random.PCG64(thermal_noise_random_seed)
            )

            E_lnb_signal = electric_fields.read_tar(
                path=os.path.join(out_dir, "lnb_input", "electric_fields.tar"),
            )
            signal_duration_s = electric_fields.get_exposure_duration_s(
                E_lnb_signal
            )

            E_lnb_noise = electric_fields.init_zeros_like_other_but_with_overhead_in_time(
                other=E_lnb_signal,
                leading_overhead_duration_s=signal_duration_s / 2,
                trailing_overhead_duration_s=signal_duration_s / 2,
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
            E_lnb_noise["electric_fields_V_per_m"] = prng.normal(
                loc=0.0,
                scale=electric_field_thermal_noise_amplitude_V_per_m,
                size=(
                    E_lnb_noise["num_antennas"],
                    E_lnb_noise["num_time_slices"],
                    3,
                ),
            )
            lnb_noise_output_power_W = signal.calculate_antenna_power(
                effective_area=telescope["lnb"]["effective_area_m2"],
                electric_field=E_lnb_noise["electric_fields_V_per_m"],
            )
            assert (
                0.9
                < (
                    telescope["lnb"]["noise_power_W"]
                    / np.mean(lnb_noise_output_power_W)
                )
                < 1.1
            )
            E_lnb_noise["electric_fields_V_per_m"] = E_lnb_noise[
                "electric_fields_V_per_m"
            ].astype(np.float32)
            electric_fields.write_tar(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                electric_fields=E_lnb_noise,
            )

            print("Done.")

    # Total electric fields leaving lnbs
    # ----------------------------------
    lnb_signal_and_noise_output_dir = os.path.join(
        out_dir, "lnb_signal_and_noise_output"
    )
    if not os.path.exists(lnb_signal_and_noise_output_dir):
        with rnw.Directory(lnb_signal_and_noise_output_dir) as tmp_dir:
            print(
                "Adding noise and signal leaving low noise block converters ... ",
                end="",
                flush=True,
            )
            E_lnb_signal = electric_fields.read_tar(
                path=os.path.join(
                    out_dir, "lnb_signal_output", "electric_fields.tar"
                )
            )

            E_lnb_noise = electric_fields.read_tar(
                path=os.path.join(
                    out_dir, "lnb_noise_output", "electric_fields.tar"
                )
            )
            E_lnb_noise_and_signal = (
                electric_fields.add_first_to_second_according_to_global_time(
                    first=E_lnb_signal,
                    second=E_lnb_noise,
                )
            )
            electric_fields.write_tar(
                path=os.path.join(tmp_dir, "electric_fields.tar"),
                electric_fields=E_lnb_noise_and_signal,
            )
            print("Done.")


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

    electric_field_leaving_feed_horns = electric_fields.init_zeros_like(
        other=electric_fields_entering_feed_horns
    )

    electric_field_leaving_feed_horns["electric_fields_V_per_m"] = (
        np.sqrt(feed_horn_gain).astype(np.float32)
        * electric_fields_entering_feed_horns["electric_fields_V_per_m"]
    )
    return electric_field_leaving_feed_horns
