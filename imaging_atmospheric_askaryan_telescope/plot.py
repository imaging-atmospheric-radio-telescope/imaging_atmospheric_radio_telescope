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


def ax_add_electric_field(
    ax,
    electric_fields,
    dimension_mask=[1, 1, 0],
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

    E = electric_fields["electric_fields"]
    E_amplitude = np.linalg.norm(E[:, :, dimension_mask], axis=2)
    E_amplitude *= amplitude_scale

    if vmin == None and vmax == None:
        E_max = np.max(E_amplitude)
        vmax = 10 ** np.ceil(np.log10(E_max))
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
    dimension_mask=[1, 1, 0],
):
    fig = plt.figure(figsize=figsize)
    # left, bottom, width, height
    ax = fig.add_axes([0.15, 0.125, 0.65, 0.8])
    ax_cmap = fig.add_axes([0.85, 0.125, 0.025, 0.8])

    im = ax_add_electric_field(
        ax=ax,
        electric_fields=electric_fields,
        dimension_mask=dimension_mask,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        time_scale=1e9,
        amplitude_scale=1e6,
    )
    gst_ns = 1e9 * electric_fields["global_start_time"]
    ax.set_title("time {:.2f}ns".format(gst_ns), loc="right")
    ax.set_xlabel("telescope time / ns")
    ax.set_ylabel("channels / 1")
    plt.colorbar(im, cax=ax_cmap)
    ax_cmap.set_ylabel("norm(electric field) / $\mu$ V m$^{-1}$")
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


def save_image_slices_power(
    power,
    time_slice_duration,
    antenna_positions,
    path,
    time_slice_region_of_interest=np.arange(70, 200),
    dpi=160,
    figsize=(12, 4),
):
    os.makedirs(path, exist_ok=True)

    powerx = power[:, :, 0] * 1e9  # in nW
    powery = power[:, :, 1] * 1e9
    powers = powerx + powery
    power_max = powers[:, :].max()

    pixel_directions_x = antenna_positions[:, 0]
    pixel_directions_y = antenna_positions[:, 1]

    for idx, time_slice in enumerate(time_slice_region_of_interest):
        fig, axarr = plt.subplots(1, 3, figsize=figsize)
        t = time_slice * time_slice_duration
        time_info = (
            "t: "
            + str(np.round(t * 1e9, 3))
            + "ns, slice: {: 6d}".format(time_slice)
        )
        fig.suptitle(time_info + "\n" + "north, west, and sum power / nW")
        add2ax(
            ax=axarr[0],
            pixel_amplitudes=powerx[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=0,
            vmax=power_max,
            colormap="viridis",
        )
        axarr[0].set_xlabel("x/m")
        axarr[0].set_ylabel("y/m")
        axarr[0].spines["right"].set_visible(False)
        axarr[0].spines["top"].set_visible(False)

        add2ax(
            ax=axarr[1],
            pixel_amplitudes=powery[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=0,
            vmax=power_max,
            colormap="viridis",
        )
        axarr[1].set_xlabel("x/m")
        axarr[1].spines["right"].set_visible(False)
        axarr[1].spines["top"].set_visible(False)
        axarr[1].yaxis.set_visible(False)

        add2ax(
            ax=axarr[2],
            pixel_amplitudes=powers[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=0,
            vmax=power_max,
            colormap="viridis",
        )
        axarr[2].set_xlabel("x/m")
        axarr[2].spines["right"].set_visible(False)
        axarr[2].spines["top"].set_visible(False)
        axarr[2].yaxis.set_visible(False)

        plt.savefig(
            os.path.join(path, "image_{:06d}.jpg".format(idx)), dpi=dpi
        )
        plt.close("all")
