import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def add2ax(
    ax, 
    pixel_amplitudes, 
    pixel_directions_x, 
    pixel_directions_y, 
    colormap='seismic', 
    hexrotation=0,
    vmin=None, 
    vmax=None,
    colorbar_label=None
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
    area = fov*fov
    bin_radius = 1.15 * np.sqrt(area/number_pixels)
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
             orientation=orientation
         )
     )
    p = PatchCollection(patches, cmap=colormap, alpha=1, edgecolor='none')
    p.set_array(A)
    p.set_clim([vmin, vmax])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(p, cax=cax)
    if colorbar_label:
        cax.set_label(colorbar_label)

    ax.add_collection(p)
    ax.set_aspect('equal')
    return p


def save_image_slices(
    out_directory,
    image_sequence,
    pixel_directions_x,
    pixel_directions_y,
    time_slice_region_of_interest=np.arange(70,200),
    colormap='seismic',
    hexrotation=0,
    xlabel='x/deg',
    ylabel='y/deg',
    colorbar_label='E/(V/m)'
):
    os.makedirs(out_directory, exist_ok=True)

    vmax = image_sequence[:,:].max()
    vmin = image_sequence[:,:].min()

    abs_max = np.max(np.abs([vmax, vmin]))

    for time_slice in time_slice_region_of_interest:
        plt.figure(figsize=(12, 6.75))
        ax = plt.gca()
        add2ax(
            ax=ax, 
            pixel_amplitudes=image_sequence[:, time_slice], 
            pixel_directions_x=pixel_directions_x, 
            pixel_directions_y=pixel_directions_y, 
            colormap=colormap, 
            hexrotation=hexrotation, 
            vmin=abs_max,
            vmax=-abs_max,
            colorbar_label=colorbar_label,
        )
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.savefig(
            os.path.join(out_directory, "image_{:06d}".format(time_slice)),
            dpi=160
        )
        plt.close('all')