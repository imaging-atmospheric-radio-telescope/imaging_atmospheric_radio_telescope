# Copyright 2017 Sebastian A. Mueller
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import subprocess
import imaging_atmospheric_askaryan_telescope as iaat
import tempfile


def ax_set_spines(ax, positions=["left", "bottom", "right", "top"]):
    for pos in ["left", "bottom", "right", "top"]:
        if pos in positions:
            ax.spines[pos].set_visible(True)
        else:
            ax.spines[pos].set_visible(False)


def make_vmax_to_match_decades(v):
    floating_v_max = np.max(v)
    vmax = 10 ** np.ceil(np.log10(floating_v_max))
    return vmax


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
):
    time_bin_edges = iaat.electric_fields.make_time_bin_edges(
        electric_fields=electric_fields, global_time=False,
    )
    antenna_bin_edges = iaat.electric_fields.make_antenna_bin_edges(
        electric_fields=electric_fields
    )

    E_amplitude = iaat.electric_fields.get_combined_norm_of_components(
        electric_fields=electric_fields, component_mask=component_mask,
    )
    E_amplitude *= amplitude_scale

    if vmin == None and vmax == None:
        vmax = make_vmax_to_match_decades(v=E_amplitude)
        vmin = vmax * 1e-3

    im = ax.pcolormesh(
        time_bin_edges * time_scale,
        antenna_bin_edges,
        E_amplitude,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    return im


def write_figure_electric_fields_overview(
    electric_fields,
    path,
    figsize=(16 / 2, 9 / 2),
    dpi=200,
    cmap="viridis",
    norm=matplotlib.colors.LogNorm(),
    vmin=None,
    vmax=None,
    component_mask=[1, 1, 0],
):
    fig = plt.figure(figsize=figsize)
    # left, bottom, width, height
    ax = fig.add_axes([0.15, 0.125, 0.65, 0.8])
    ax_cmap = fig.add_axes([0.85, 0.125, 0.025, 0.8])

    im = ax_add_electric_field(
        ax=ax,
        electric_fields=electric_fields,
        component_mask=component_mask,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        time_scale=1e9,
        amplitude_scale=1e6,
    )
    gst_ns = 1e9 * electric_fields["global_start_time"]
    ax.set_title("absolute time: {:.2f}ns".format(gst_ns), loc="right")
    ax.set_xlabel("relative time / ns")
    ax.set_ylabel("channels / 1")
    plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel("norm(electric field) / $\mu$ V m$^{-1}$")
    fig.savefig(path, dpi=dpi)
    plt.close("all")


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
    vmin=None,
    vmax=None,
    figsize=(16 / 2, 9 / 2),
    dpi=200,
    title=None,
):
    fig = plt.figure(figsize=figsize)
    # left, bottom, width, height
    ax = fig.add_axes([0.15, 0.125, 0.65, 0.8])
    ax.set_title(title)
    ax_cmap = fig.add_axes([0.85, 0.125, 0.025, 0.8])
    im = ax.pcolormesh(
        x_bin_edges,
        y_bin_edges,
        matrix,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel(z_label)
    if cmap_marker:
        ax_cmap.plot(ax_cmap.get_xlim(), [cmap_marker, cmap_marker], "r-")

    fig.savefig(path, dpi=dpi)
    plt.close("all")


def write_figure_electric_fields_power_density_spectrum(
    path,
    electric_fields,
    component_mask,
    num_time_slices_to_average_over,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    figsize=(16 / 2, 9 / 2),
    dpi=200,
    vim_fraction_of_vmax=1e-4,
):
    if norm == None:
        norm = matplotlib.colors.LogNorm()

    ef = electric_fields

    pds = np.zeros(
        shape=(ef["num_antennas"], 1 + (num_time_slices_to_average_over // 2))
    )

    E_amplitude = iaat.electric_fields.get_combined_norm_of_components(
        electric_fields=electric_fields, component_mask=component_mask,
    )

    _last_f_antenna_bin_edges = None
    for antenna in range(ef["num_antennas"]):
        (
            f_antenna_bin_edges,
            pds_antenna,
        ) = iaat.signal.estimate_power_spectrum_density(
            amplitudes=E_amplitude[antenna, :],
            time_slice_duration=ef["time_slice_duration"],
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

    exposure_time = ef["time_slice_duration"] * ef["electric_fields"].shape[1]
    sampling_frequency = 1.0 / ef["time_slice_duration"]

    if vmin == None and vmax == None:
        vmax = make_vmax_to_match_decades(v=pds)
        vmin = vmax * vim_fraction_of_vmax

    write_matrix(
        path=path,
        matrix=pds,
        x_bin_edges=f_antenna_bin_edges * 1e-9,
        y_bin_edges=antenna_bin_edges,
        x_label="frequency / GHz",
        y_label="channels / 1",
        z_label="power spectrum density / V$^{2}$ m$^{-2}$ Hz$^{-1}$",
        norm=norm,
        cmap=cmap,
        title="exposure time: {:.1f}ns, sampling frequency: {:.1f}GHz".format(
            1e9 * exposure_time, 1e-9 * sampling_frequency
        ),
        figsize=figsize,
        dpi=dpi,
        vmin=vmin,
        vmax=vmax,
    )


def write_figure_lnb_power(
    path,
    lnb_power,
    antenna_bin_edges,
    relative_time_bin_edges,
    global_start_time,
    expected_noise_power=None,
    norm=None,
    vmin=None,
    vmax=None,
    vim_fraction_of_vmax=1e-3,
    dpi=200,
):
    lnb_power_pW = 1e12 * lnb_power

    if norm == None:
        norm = matplotlib.colors.LogNorm()

    if vmin == None and vmax == None:
        vmax = make_vmax_to_match_decades(v=lnb_power_pW)
        vmin = vmax * vim_fraction_of_vmax

    if expected_noise_power:
        expected_noise_power_pW = 1e12 * expected_noise_power
    else:
        expected_noise_power_pW = None

    write_matrix(
        path=path,
        matrix=lnb_power_pW,
        x_bin_edges=1e9 * relative_time_bin_edges,
        y_bin_edges=antenna_bin_edges,
        x_label="relative time / ns",
        y_label="channels / 1",
        z_label="power / pW",
        title="absolute time: {:.2f}ns".format(1e9 * global_start_time),
        norm=norm,
        vmax=vmax,
        vmin=vmin,
        dpi=dpi,
        cmap_marker=expected_noise_power_pW,
    )


def write_figure_antenna_positions(
    positions, path, figsize=(16 / 2, 9 / 2), dpi=200,
):
    fig = plt.figure(figsize=figsize)
    # left, bottom, width, height
    ax = fig.add_axes([0.15, 0.125, 0.8, 0.8])

    r = np.hypot(positions[:, 0], positions[:, 1])
    rmax = np.max(r) * 1.1
    ax.plot(positions[:, 0], positions[:, 1], "xk")

    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, rmax])
    ax.set_aspect("equal")

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    fig.savefig(path, dpi=dpi)
    plt.close("all")


def write_figure_gain(
    path, frequency, gain, figsize=(16 / 2, 9 / 2), dpi=200,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.125, 0.8, 0.8])
    frequency_GHz = frequency / 1e9

    fmin = 10 ** np.floor(np.log10(np.min(frequency_GHz)))
    fmax = 10 ** np.ceil(np.log10(np.max(frequency_GHz)))

    gmin = 10 ** np.floor(np.log10(np.min(gain)))
    gmax = 10 ** np.ceil(np.log10(np.max(gain)))

    ax.plot(frequency_GHz, gain, "-k")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

    ax.set_xlim([fmin, fmax])
    ax.set_ylim([gmin, gmax])

    ax.loglog()
    ax.set_ylabel("gain / 1")
    ax.set_xlabel("frequency / GHz")
    fig.savefig(path, dpi=dpi)
    plt.close("all")


def add2ax(
    ax,
    pixel_amplitudes,
    pixel_directions_x,
    pixel_directions_y,
    colormap="seismic",
    hexrotation=0,
    vmin=None,
    vmax=None,
):
    A = pixel_amplitudes
    dx = pixel_directions_x
    dy = pixel_directions_y
    number_pixels = A.shape[0]

    if vmin is None:
        vmin = A.min()
    if vmax is None:
        vmax = A.max()

    fov = np.abs(dx).max() * 1.05
    area = fov * fov
    bin_radius = 1.15 * np.sqrt(area / number_pixels)
    nfov = fov + bin_radius
    ax.set_xlim([-nfov, nfov])
    ax.set_ylim([-nfov, nfov])
    ax.set_aspect("equal")
    orientation = hexrotation
    patches = []
    for d in range(number_pixels):
        patches.append(
            RegularPolygon(
                (dx[d], dy[d]),
                numVertices=6,
                radius=bin_radius,
                orientation=orientation,
            )
        )
    p = PatchCollection(patches, cmap=colormap, alpha=1, edgecolor="none")
    p.set_array(A)
    p.set_clim([vmin, vmax])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(p, cax=cax)

    ax.add_collection(p)
    ax.set_aspect("equal")
    return p


def make_video_from_image_slices(
    image_slice_dir,
    out_path,
    image_slice_filename_wildcard="image_%06d.png",
    fps=15,
    threads=1,
):
    """
    Saves a H264 1080p video of a 3D rendering of a photon-stream event to the
    outputpath. In the 3D rendering the event is rotated 360 degrees around the
    optical axis of the telescope.
    """
    avconv_command = [
        "ffmpeg",
        "-y",  # force overwriting of existing output file
        "-framerate",
        str(int(fps)),  # Frames per second
        "-f",
        "image2",
        "-i",
        os.path.join(image_slice_dir, image_slice_filename_wildcard),
        "-c:v",
        "h264",
        # '-s', '1920x1080',  # sample images to FullHD 1080p
        "-crf",
        "23",  # high quality 0 (best) to 53 (worst)
        "-crf_max",
        "25",  # worst quality allowed
        "-threads",
        str(threads),
        os.path.splitext(out_path)[0] + ".mov",
    ]
    subprocess.call(avconv_command)


ELECTRON_VOLT_J = 1.602176634e-19


def save_image_slices_energy_deposite(
    readout_energy,
    readout_time_slice_duration,
    antenna_positions,
    path,
    global_start_time=0.0,
    dpi=160,
    figsize=(12, 4),
):
    os.makedirs(path, exist_ok=True)

    enex = readout_energy[:, :, 0] / ELECTRON_VOLT_J  # in eV
    eney = readout_energy[:, :, 1] / ELECTRON_VOLT_J  # in eV
    enes = enex + eney
    energy_max = enes[:, :].max()

    pixel_directions_x = antenna_positions[:, 0]
    pixel_directions_y = antenna_positions[:, 1]
    num_readout_time_slices = readout_energy.shape[1]

    for readout_time_slice in range(num_readout_time_slices):
        fig, axarr = plt.subplots(1, 3, figsize=figsize)
        relative_time = readout_time_slice_duration * readout_time_slice
        time_info = (
            "absolute time: {: 6.3f}ns, ".format(1e9 * global_start_time)
            + "relative time: {: 8.1f}ns, ".format(1e9 * relative_time)
            + "readout-time-slice: {: 6d}".format(readout_time_slice)
        )
        fig.suptitle(
            time_info + "\n" + "(North, West, Sum) deposited energy / eV"
        )
        add2ax(
            ax=axarr[0],
            pixel_amplitudes=enex[:, readout_time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=0,
            vmax=energy_max,
            colormap="viridis",
        )
        axarr[0].set_xlabel("x/m")
        axarr[0].set_ylabel("y/m")
        axarr[0].spines["right"].set_visible(False)
        axarr[0].spines["top"].set_visible(False)

        add2ax(
            ax=axarr[1],
            pixel_amplitudes=eney[:, readout_time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=0,
            vmax=energy_max,
            colormap="viridis",
        )
        axarr[1].set_xlabel("x/m")
        axarr[1].spines["right"].set_visible(False)
        axarr[1].spines["top"].set_visible(False)
        axarr[1].yaxis.set_visible(False)

        add2ax(
            ax=axarr[2],
            pixel_amplitudes=enes[:, readout_time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=0,
            vmax=energy_max,
            colormap="viridis",
        )
        axarr[2].set_xlabel("x/m")
        axarr[2].spines["right"].set_visible(False)
        axarr[2].spines["top"].set_visible(False)
        axarr[2].yaxis.set_visible(False)

        plt.savefig(
            os.path.join(path, "image_{:06d}.jpg".format(readout_time_slice)),
            dpi=dpi,
        )
        plt.close("all")
