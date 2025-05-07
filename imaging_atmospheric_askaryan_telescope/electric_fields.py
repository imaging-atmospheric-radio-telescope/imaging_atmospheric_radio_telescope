import numpy as np
import os
import copy
from . import corsika
from . import signal
from . import time_series


def init_from_coreas_electric_fields(coreas_electric_fields):
    raw = coreas_electric_fields
    time_slice_duration_s = (
        corsika.coreas.coreas_electric_fields.estimate_time_slice_duration_s(
            coreas_electric_fields=raw
        )
    )
    corsika.coreas.coreas_electric_fields.assert_same_time_slice_duration(
        coreas_electric_fields=raw,
        time_slice_duration_s=time_slice_duration_s,
    )
    num_antennas = len(raw)
    num_time_slices = raw[0].shape[0]

    global_start_time_s = np.min(
        [raw[a]["time_s"] for a in range(num_antennas)]
    )

    start_time_offsets_s = np.array(
        [
            raw[a]["time_s"][0] - global_start_time_s
            for a in range(num_antennas)
        ]
    )

    start_slice_offsets_s = np.round(
        start_time_offsets_s / time_slice_duration_s
    ).astype(np.int64)
    assert np.all(start_slice_offsets_s == 0)

    E = time_series.zeros(
        time_slice_duration_s=time_slice_duration_s,
        num_time_slices=num_time_slices,
        num_channels=num_antennas,
        num_components=3,
        global_start_time_s=global_start_time_s,
        si_unit="V_per_m",
        dtype="float32",
    )

    CGS_TO_SI = (
        corsika.coreas.coreas_electric_fields.CGS_statVolt_per_cm_to_SI_Volt_per_meter
    )
    for a in range(num_antennas):
        E[a, :, 0] = raw[a]["E_north_statVolt_per_cm"] * CGS_TO_SI
        E[a, :, 1] = raw[a]["E_west_statVolt_per_cm"] * CGS_TO_SI
        E[a, :, 2] = raw[a]["E_vertical_statVolt_per_cm"] * CGS_TO_SI

    return E


def to_coreas_electric_fields(electric_fields):
    ef = electric_fields
    assert ef.si_unit == "V_per_m"
    assert ef.num_components == 3

    raw = corsika.coreas.coreas_electric_fields.init_zeros(
        num_antennas=ef.num_channels,
        num_time_slices=ef.num_time_slices,
    )
    CGS_TO_SI = (
        corsika.coreas.coreas_electric_fields.CGS_statVolt_per_cm_to_SI_Volt_per_meter
    )

    for a in range(ef.num_channels):
        raw[a]["time_s"] = ef.global_start_time_s + np.linspace(
            0,
            ef.time_slice_duration_s * ef.num_time_slices,
            ef.num_time_slices,
        )

        raw[a]["E_north_statVolt_per_cm"] = ef[a, :, 0] / CGS_TO_SI
        raw[a]["E_west_statVolt_per_cm"] = ef[a, :, 1] / CGS_TO_SI
        raw[a]["E_vertical_statVolt_per_cm"] = ef[a, :, 2] / CGS_TO_SI

    return raw


def integrate_power_over_time(
    electric_fields,
    channel_effective_area_m2,
    component_mask=None,
):
    E_magnitude_V_per_m = electric_fields.norm_components(
        component_mask=component_mask
    )
    P_W = signal.calculate_antenna_power_W(
        effective_area_m2=channel_effective_area_m2,
        electric_field_V_per_m=E_magnitude_V_per_m[:],
    )
    Ene_J = np.sum(P_W, axis=1) * electric_fields.time_slice_duration_s
    return Ene_J


def estimate_power_spectrum_density_W_per_Hz_per_m2(
    electric_fields,
    antenna_effective_area_m2,
    frequency_bin_edges_Hz,
    components=[True, True, True],
):
    E = electric_fields
    assert E.si_unit == "V_per_m"

    nu_bin_edges = frequency_bin_edges_Hz
    nu_num_nins = len(nu_bin_edges) - 1

    mat_W_per_Hz_per_m2 = np.zeros(shape=(nu_num_nins, E.num_channels))

    for antenna in range(E.num_channels):
        for component in [0, 1, 2]:
            if components[component]:
                _e_by_nu = signal.split_into_frequency_bins(
                    amplitudes=E[antenna, :, component],
                    time_slice_duration_s=E.time_slice_duration_s,
                    frequency_bin_edges_Hz=nu_bin_edges,
                )
                for nu in range(nu_num_nins):
                    nu_bandwidth_Hz = nu_bin_edges[nu + 1] - nu_bin_edges[nu]
                    _Power_W = np.mean(
                        signal.calculate_antenna_power_W(
                            effective_area_m2=antenna_effective_area_m2,
                            electric_field_V_per_m=_e_by_nu[nu],
                        )
                    )
                    mat_W_per_Hz_per_m2[nu, antenna] += (
                        _Power_W / nu_bandwidth_Hz / antenna_effective_area_m2
                    )

    return mat_W_per_Hz_per_m2
