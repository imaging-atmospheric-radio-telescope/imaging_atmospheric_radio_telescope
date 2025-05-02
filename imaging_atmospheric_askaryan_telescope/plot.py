import sebastians_matplotlib_addons as seb
import imaging_atmospheric_askaryan_telescope as iaat
import tempfile
import os
import numpy as np
import subprocess
import json_numpy


CONFIG = {
    "matplotlib": {"mathtext.fontset": "cm", "font.family": "STIXGeneral"},
    "particle_colors": {
        "gamma": "black",
        "electron": "blue",
        "proton": "red",
        "helium": "orange",
    },
}

seb.matplotlib.rcParams.update(CONFIG["matplotlib"])


FIG_2560X1080F1P5 = {"rows": 1080, "cols": 2560, "fontsize": 1.5}
FIG_512X1080F1P5 = {"rows": 1080, "cols": 512, "fontsize": 1.5}
FIG_1080X1080F1P5 = {"rows": 1080, "cols": 1080, "fontsize": 1.5}
FIG_1920X1080F1P5 = {"rows": 1080, "cols": 1920, "fontsize": 1.5}
FIG_3840X1080F1P5 = {"rows": 2160, "cols": 3840, "fontsize": 3.0}


def log10_limits(v, ratio=0):
    vmin = (1 - ratio) * 10 ** np.floor(np.log10(np.min(v)))
    vmax = (1 + ratio) * 10 ** np.ceil(np.log10(np.max(v)))
    return vmin, vmax


def make_vmax_to_match_decades(v):
    floating_v_max = np.max(v)
    vmax = 10 ** np.ceil(np.log10(floating_v_max))
    return vmax


def write_figure_gain(
    path,
    frequency,
    gain,
    scale="G",
    figsize={"rows": 1080, "cols": 1440, "fontsize": 1.5},
):
    if scale == None:
        s = 1
        s_str = ""
    elif scale == "M":
        s = 1e6
        s_str = scale
    elif scale == "G":
        s = 1e9
        s_str = scale

    fig = seb.figure(style=figsize)
    ax = seb.add_axes(fig=fig, span=[0.15, 0.2, 0.8, 0.75])

    frequency_scale = frequency / s

    fmin, fmax = log10_limits(v=frequency_scale, ratio=0.0)
    gmin, gmax = log10_limits(v=gain, ratio=0.05)

    ax.plot(frequency_scale, gain, "-k")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1, which="both")

    ax.set_xlim([fmin, fmax])
    ax.set_ylim([gmin, gmax])

    ax.loglog()
    ax.set_ylabel("gain / 1")
    ax.set_xlabel("frequency / {:s}Hz".format(s_str))
    fig.savefig(path)
    seb.close(fig)


def ax_add_hexagonal_pixels(
    ax,
    v,
    x,
    y,
    cmap="viridis",
    hexrotation=0,
    vmin=None,
    vmax=None,
    alpha=1,
):
    num_pixels = v.shape[0]

    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()

    fov = np.abs(x).max() * 1.05
    area = fov * fov
    bin_radius = 1.15 * np.sqrt(area / num_pixels)
    nfov = fov + bin_radius
    ax.set_xlim([-nfov, nfov])
    ax.set_ylim([-nfov, nfov])
    ax.set_aspect("equal")
    orientation = hexrotation
    patches = []
    for d in range(num_pixels):
        patches.append(
            seb.matplotlib.patches.RegularPolygon(
                (x[d], y[d]),
                numVertices=6,
                radius=bin_radius,
                orientation=orientation,
            )
        )
    p = seb.matplotlib.collections.PatchCollection(
        patches,
        cmap=cmap,
        alpha=alpha,
        edgecolor="none",
    )
    p.set_array(v)
    p.set_clim([vmin, vmax])
    ax.add_collection(p)
    ax.set_aspect("equal")
    return p


def save_image_slices_energy_deposite(
    readout_energy_J,
    readout_time_slice_duration_s,
    antenna_positions,
    path,
    global_start_time_s=0.0,
    units=None,
    bandwidth_Hz=None,
    mirror_area_m2=None,
    cmap="viridis",
    image_x_label="x",
    image_y_label="y",
    rows=1024,
):
    os.makedirs(path, exist_ok=True)

    if units == None:
        enex = readout_energy_J[:, :, 0]
        eney = readout_energy_J[:, :, 1]
        estr = "deposited energy / J"
    elif units == "electron_volt":
        enex = (
            readout_energy_J[:, :, 0] / iaat.signal.ELECTRON_VOLT_J * 1e6
        )  # ueV
        eney = (
            readout_energy_J[:, :, 1] / iaat.signal.ELECTRON_VOLT_J * 1e6
        )  # ueV
        estr = r"deposited energy / $\mu$eV"
    elif units == "black_body_temperature":
        assert bandwidth_Hz != None
        px_W = readout_energy_J[:, :, 0] / readout_time_slice_duration_s  # W
        py_W = readout_energy_J[:, :, 1] / readout_time_slice_duration_s  # W
        enex = iaat.signal.radiated_power_to_blackbody_temperature(
            power_W=px_W,
            bandwidth_Hz=bandwidth_Hz,
        )
        eney = iaat.signal.radiated_power_to_blackbody_temperature(
            power_W=py_W,
            bandwidth_Hz=bandwidth_Hz,
        )
        estr = "black body temperature / K"
    elif units == "jansky":
        assert bandwidth_Hz != None
        assert mirror_area_m2 != None
        px_W = readout_energy_J[:, :, 0] / readout_time_slice_duration_s  # W
        py_W = readout_energy_J[:, :, 1] / readout_time_slice_duration_s  # W
        _per_area_per_bandwidth = mirror_area_m2 ** (-1) * bandwidth_Hz ** (-1)
        flux_density_x_W_per_m2_per_Hz = px_W * _per_area_per_bandwidth
        flux_density_y_W_per_m2_per_Hz = py_W * _per_area_per_bandwidth
        flux_density_x_Jy = 1e26 * flux_density_x_W_per_m2_per_Hz
        flux_density_y_Jy = 1e26 * flux_density_y_W_per_m2_per_Hz
        enex = flux_density_x_Jy
        eney = flux_density_y_Jy
        estr = "flux density / Jy"
    else:
        raise AttributeError("Units not known.")

    enes = enex + eney
    energy_max = enes[:, :].max()

    pixel_directions_x = antenna_positions[:, 0]
    pixel_directions_y = antenna_positions[:, 1]
    num_readout_time_slices = readout_energy_J.shape[1]

    CB = {"rows": rows, "cols": rows // 2, "fontsize": 2.0}

    fig = seb.figure(style=CB)
    ax_cb = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.3, 0.85])
    _m = seb.plt.cm.ScalarMappable(norm=None, cmap=cmap)
    _m.set_clim([0.0, energy_max])
    seb.plt.colorbar(_m, cax=ax_cb, label=estr)
    fig.savefig(
        os.path.join(path, "colormap.jpg"),
    )
    seb.close(fig)

    with open(os.path.join(path, "time.json"), "wt") as f:
        f.write(
            json_numpy.dumps(
                {
                    "global_start_time_s": global_start_time_s,
                    "readout_time_slice_duration_s": readout_time_slice_duration_s,
                }
            )
        )

    IMG = {"rows": rows, "cols": rows, "fontsize": 1.0}
    IMGS = {"rows": 2 * rows, "cols": 2 * rows, "fontsize": 1.0}

    for readout_time_slice in range(num_readout_time_slices):
        fig = seb.figure(style=IMG)
        ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.85, 0.85])
        ax_add_hexagonal_pixels(
            ax=ax,
            v=enex[:, readout_time_slice],
            x=pixel_directions_x,
            y=pixel_directions_y,
            vmin=0.0,
            vmax=energy_max,
            cmap=cmap,
        )
        fig.savefig(
            os.path.join(
                path, "horizontal_{:06d}_.jpg".format(readout_time_slice)
            ),
        )
        ax.set_xlabel(image_x_label)
        ax.set_ylabel(image_y_label)
        seb.close(fig)

        fig = seb.figure(style=IMG)
        ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.85, 0.85])
        ax_add_hexagonal_pixels(
            ax=ax,
            v=eney[:, readout_time_slice],
            x=pixel_directions_x,
            y=pixel_directions_y,
            vmin=0.0,
            vmax=energy_max,
            cmap=cmap,
        )
        fig.savefig(
            os.path.join(
                path, "vertical_{:06d}.jpg".format(readout_time_slice)
            ),
        )
        ax.set_xlabel(image_x_label)
        ax.set_ylabel(image_y_label)
        seb.close(fig)

        fig = seb.figure(style=IMGS)
        ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.85, 0.85])
        ax_add_hexagonal_pixels(
            ax=ax,
            v=enes[:, readout_time_slice],
            x=pixel_directions_x,
            y=pixel_directions_y,
            vmin=0.0,
            vmax=energy_max,
            cmap=cmap,
        )
        fig.savefig(
            os.path.join(path, "sum_{:06d}.jpg".format(readout_time_slice)),
        )
        ax.set_xlabel(image_x_label)
        ax.set_ylabel(image_y_label)
        seb.close(fig)


def write_matrix(
    path,
    matrix,
    x_bin_edges,
    y_bin_edges,
    x_label,
    y_label,
    z_label,
    cmap="viridis",
    cmap_marker=None,
    norm=None,
    figsize=FIG_1920X1080F1P5,
    title=None,
):
    fig = seb.figure(style=figsize)
    ax = seb.add_axes(fig=fig, span=[0.125, 0.15, 0.68, 0.75])
    ax_cmap = seb.add_axes(fig=fig, span=[0.83, 0.15, 0.025, 0.75])
    ax.set_title(title, fontsize="small")
    im = ax.pcolormesh(
        x_bin_edges,
        y_bin_edges,
        matrix,
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    seb.plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel(z_label)
    if cmap_marker:
        ax_cmap.plot(ax_cmap.get_xlim(), [cmap_marker, cmap_marker], "r-")

    fig.savefig(path)
    seb.close(fig)


def write_figure_electric_fields_power_density_spectrum(
    path,
    electric_fields,
    component_mask,
    num_time_slices_to_average_over,
    channels_label="channels / 1",
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    vim_fraction_of_vmax=1e-3,
    figsize=FIG_1920X1080F1P5,
    roi_frequency=None,
):
    if norm == None:
        norm = seb.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    ef = electric_fields

    pds = np.zeros(
        shape=(ef["num_antennas"], 1 + (num_time_slices_to_average_over // 2))
    )

    E_amplitude = iaat.electric_fields.get_combined_norm_of_components(
        electric_fields=electric_fields,
        component_mask=component_mask,
    )

    _last_f_antenna_bin_edges = None
    for antenna in range(ef["num_antennas"]):
        (
            f_antenna_bin_edges,
            pds_antenna,
        ) = iaat.signal.estimate_power_spectrum_density(
            amplitudes=E_amplitude[antenna, :],
            time_slice_duration_s=ef["time_slice_duration_s"],
            num_time_slices_to_average_over=num_time_slices_to_average_over,
        )
        pds[antenna, :] = pds_antenna
        if _last_f_antenna_bin_edges is None:
            _last_f_antenna_bin_edges = f_antenna_bin_edges
        else:
            np.testing.assert_array_almost_equal(
                _last_f_antenna_bin_edges, f_antenna_bin_edges
            )

    antenna_bin_edges = iaat.electric_fields.make_antenna_bin_edges(
        electric_fields=ef
    )

    exposure_time_s = (
        ef["time_slice_duration_s"] * ef["electric_fields_V_per_m"].shape[1]
    )
    sampling_frequency_Hz = 1.0 / ef["time_slice_duration_s"]

    if vmin == None and vmax == None:
        vmax = make_vmax_to_match_decades(v=pds)
        vmin = vmax * vim_fraction_of_vmax

    if roi_frequency == None:
        roi_frequency = [f_antenna_bin_edges[0], f_antenna_bin_edges[-1]]

    start_f_slice = np.argmin(np.abs(f_antenna_bin_edges - roi_frequency[0]))
    stop_f_slice = np.argmin(np.abs(f_antenna_bin_edges - roi_frequency[1]))

    write_matrix(
        path=path,
        matrix=pds[:, start_f_slice : stop_f_slice - 1],
        x_bin_edges=f_antenna_bin_edges[start_f_slice:stop_f_slice] * 1e-9,
        y_bin_edges=antenna_bin_edges,
        x_label="frequency / GHz",
        y_label=channels_label,
        z_label="power spectrum density / V$^{2}$ m$^{-2}$ Hz$^{-1}$",
        norm=norm,
        cmap=cmap,
        title="exposure time: {:.1f}ns, sampling rate: {:.1f}Gsps".format(
            1e9 * exposure_time_s, 1e-9 * sampling_frequency_Hz
        ),
        figsize=figsize,
    )


def ax_add_electric_field(
    ax,
    electric_fields,
    component_mask=[1, 1, 0],
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    time_scale=1,
    amplitude_scale=1,
    roi_time=None,
):
    time_bin_edges = iaat.electric_fields.make_time_bin_edges(
        electric_fields=electric_fields,
        global_time=False,
    )
    antenna_bin_edges = iaat.electric_fields.make_antenna_bin_edges(
        electric_fields=electric_fields
    )

    E_amplitude = iaat.electric_fields.get_combined_norm_of_components(
        electric_fields=electric_fields,
        component_mask=component_mask,
    )

    if vmin == None and vmax == None:
        vmax = make_vmax_to_match_decades(v=E_amplitude)
        vmin = vmax * 1e-3

    if norm is None:
        norm = seb.matplotlib.colors.LogNorm(
            vmin=vmin * amplitude_scale, vmax=vmax * amplitude_scale
        )

    if roi_time == None:
        start_time = 0.0
        stop_time = (
            electric_fields["num_time_slices"]
            * electric_fields["time_slice_duration_s"]
        )
        roi_time = [start_time, stop_time]

    start_time_slice = int(
        roi_time[0] / electric_fields["time_slice_duration_s"]
    )
    stop_time_slice = int(
        roi_time[1] / electric_fields["time_slice_duration_s"]
    )

    im = ax.pcolormesh(
        time_bin_edges[start_time_slice : stop_time_slice + 1] * time_scale,
        antenna_bin_edges,
        E_amplitude[:, start_time_slice:stop_time_slice] * amplitude_scale,
        cmap=cmap,
        norm=norm,
    )
    return im


def write_figure_electric_fields_overview(
    electric_fields,
    path,
    figsize=FIG_3840X1080F1P5,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    component_mask=[1, 1, 0],
    channels_label="channels / 1",
    roi_time=None,
):
    if norm is None:
        norm = seb.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    fig = seb.figure(style=figsize)
    ax = seb.add_axes(fig=fig, span=[0.125, 0.15, 0.68, 0.75])
    ax_cmap = seb.add_axes(fig=fig, span=[0.83, 0.15, 0.025, 0.75])
    im = ax_add_electric_field(
        ax=ax,
        electric_fields=electric_fields,
        component_mask=component_mask,
        cmap=cmap,
        norm=norm,
        time_scale=1e9,
        amplitude_scale=1e6,
        roi_time=roi_time,
    )
    gst_ns = 1e9 * electric_fields["global_start_time_s"]
    ax.set_xlabel("relative time / ns (absolute: {:.2f}ns)".format(gst_ns))
    ax.set_ylabel(channels_label)
    seb.plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel(r"$\vert$ electric field $\vert_2$ / $\mu$ V m$^{-1}$")
    fig.savefig(path)
    seb.close(fig)


def write_figure_lnb_power(
    path,
    lnb_power_W,
    channels_bin_edges,
    relative_time_bin_edges_s,
    global_start_time_s,
    expected_noise_power_W=None,
    norm=None,
    lnb_power_min_W=None,
    lnb_power_max_W=None,
    channels_label="channels / 1",
    figsize=FIG_3840X1080F1P5,
    lnb_power_min_fraction_of_max=1e-3,
    roi_time=None,
):
    if lnb_power_min_W == None and lnb_power_max_W == None:
        lnb_power_max_W = make_vmax_to_match_decades(v=lnb_power_W)
        lnb_power_min_W = lnb_power_max_W * lnb_power_min_fraction_of_max

    if expected_noise_power_W:
        expected_noise_power_pW = 1e12 * expected_noise_power_W
    else:
        expected_noise_power_pW = None

    if roi_time == None:
        start_time = relative_time_bin_edges_s[0]
        stop_time = relative_time_bin_edges_s[-1]
        roi_time = [start_time, stop_time]

    start_time_slice = np.argmin(
        np.abs(relative_time_bin_edges_s - roi_time[0])
    )
    stop_time_slice = np.argmin(
        np.abs(relative_time_bin_edges_s - roi_time[1])
    )

    if norm is None:
        norm = seb.matplotlib.colors.LogNorm(
            vmin=1e12 * lnb_power_min_W, vmax=1e12 * lnb_power_max_W
        )

    write_matrix(
        path=path,
        matrix=1e12 * lnb_power_W[:, start_time_slice : stop_time_slice - 1],
        x_bin_edges=1e9
        * relative_time_bin_edges_s[start_time_slice:stop_time_slice],
        y_bin_edges=channels_bin_edges,
        x_label="relative time / ns (absolute: {:.2f}ns)".format(
            1e9 * global_start_time_s
        ),
        y_label=channels_label,
        z_label="power / pW",
        norm=norm,
        figsize=figsize,
        cmap_marker=expected_noise_power_pW,
    )


def write_figure_antenna_positions(
    positions,
    path,
    figsize={"rows": 2160, "cols": 3840, "fontsize": 3.0},
):
    fig = seb.figure(style=figsize)
    ax = seb.add_axes(fig=fig, span=[0.15, 0.125, 0.8, 0.8])

    r = np.hypot(positions[:, 0], positions[:, 1])
    rmax = np.max(r) * 1.1
    ax.plot(positions[:, 0], positions[:, 1], ".k")

    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, rmax])
    ax.set_aspect("equal")

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    fig.savefig(path)
    seb.close(fig)
