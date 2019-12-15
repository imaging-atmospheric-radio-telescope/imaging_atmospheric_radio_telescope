# Copyright 2017 Sebastian A. Mueller
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import subprocess
import imaging_atmospheric_askaryan_telescope as iaat
import tempfile


def add2ax(
    ax,
    pixel_amplitudes,
    pixel_directions_x,
    pixel_directions_y,
    colormap='seismic',
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
                orientation=orientation))
    p = PatchCollection(patches, cmap=colormap, alpha=1, edgecolor='none')
    p.set_array(A)
    p.set_clim([vmin, vmax])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    plt.colorbar(p, cax=cax)

    ax.add_collection(p)
    ax.set_aspect('equal')
    return p


def save_image_slices(
    event,
    path,
    time_slice_region_of_interest=np.arange(70, 200),
    dpi=160,
    figsize=(12, 4),
):
    os.makedirs(path, exist_ok=True)

    north = event.raw_image_sensor_response.north*1e6  # in uV/m
    west = event.raw_image_sensor_response.west*1e6
    vertical = event.raw_image_sensor_response.vertical*1e6

    north_max = north[:, :].max()
    north_min = north[:, :].min()
    abs_north = np.max(np.abs([north_max, north_min]))

    west_max = west[:, :].max()
    west_min = west[:, :].min()
    abs_west = np.max(np.abs([west_max, west_min]))

    vertical_max = vertical[:, :].max()
    vertical_min = vertical[:, :].min()
    abs_vertical = np.max(np.abs([vertical_max, vertical_min]))

    pixel_directions_x = np.rad2deg(event.image_sensor.pixel_directions[:, 0])
    pixel_directions_y = np.rad2deg(event.image_sensor.pixel_directions[:, 1])

    for idx, time_slice in enumerate(time_slice_region_of_interest):
        fig, axarr = plt.subplots(1, 3, figsize=figsize)
        t = time_slice*event.simulation_truth['time_slice_duration']
        time_info = 't: ' + str(np.round(t*1e9, 1)) + 'ns'
        fig.suptitle(
            simulation_truth_info_string(event.simulation_truth) + ', ' +
            time_info + '\n' +
            'north, west, and vertical electric-field /(uV/m)')
        add2ax(
            ax=axarr[0],
            pixel_amplitudes=north[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=abs_north,
            vmax=-abs_north,)
        axarr[0].set_xlabel('x/deg')
        axarr[0].set_ylabel('y/deg')
        axarr[0].spines['right'].set_visible(False)
        axarr[0].spines['top'].set_visible(False)
        add2ax(
            ax=axarr[1],
            pixel_amplitudes=west[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=abs_west,
            vmax=-abs_west,)
        axarr[1].set_xlabel('x/deg')
        axarr[1].spines['right'].set_visible(False)
        axarr[1].spines['top'].set_visible(False)
        axarr[1].yaxis.set_visible(False)
        add2ax(
            ax=axarr[2],
            pixel_amplitudes=vertical[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            vmin=abs_vertical,
            vmax=-abs_vertical,)
        axarr[2].set_xlabel('x/deg')
        axarr[2].spines['right'].set_visible(False)
        axarr[2].spines['top'].set_visible(False)
        axarr[2].yaxis.set_visible(False)
        plt.savefig(
            os.path.join(path, "image_{:06d}".format(idx)),
            dpi=dpi)
        plt.close('all')


def simulation_truth_info_string(simulation_truth):
    st = simulation_truth
    s = ''
    if st['primary_particle_id'] == 1:
        s += 'Gamma'
    if st['primary_particle_id'] == 14:
        s += 'Proton'
    s += ', '
    s += 'E: '+str(np.round(st['energy'])) + 'GeV, '
    s += 'Az: '+str(np.round(np.rad2deg(st['azimuth']), 1)) + 'deg, '
    s += 'Zd: '+str(np.round(np.rad2deg(st['zenith_distance']), 1)) + 'deg, '
    s += 'Core North: ' + str(
            np.round(st['core_position_on_observation_level_north'], 1)
        ) + 'm, '
    s += 'Core West: ' + str(
            np.round(st['core_position_on_observation_level_west'], 1)
        ) + 'm, '
    s += 'Obs. altitude: ' + str(
            np.round(st['observation_level_altitude'])
        ) + 'm a.s.l'
    return s


def make_video_from_image_slices(
    image_slice_dir,
    out_path,
    image_slice_filename_wildcard='image_%06d.png',
    fps=15,
    threads=1,
):
    """
    Saves a H264 1080p video of a 3D rendering of a photon-stream event to the
    outputpath. In the 3D rendering the event is rotated 360 degrees around the
    optical axis of the telescope.
    """
    avconv_command = [
        'ffmpeg',
        '-y',  # force overwriting of existing output file
        '-framerate', str(int(fps)),  # Frames per second
        '-f', 'image2',
        '-i', os.path.join(image_slice_dir, image_slice_filename_wildcard),
        '-c:v', 'h264',
        # '-s', '1920x1080',  # sample images to FullHD 1080p
        '-crf', '23',  # high quality 0 (best) to 53 (worst)
        '-crf_max', '25',  # worst quality allowed
        '-threads', str(threads),
        os.path.splitext(out_path)[0] + '.mov']
    subprocess.call(avconv_command)


def save_total_energy_deposite(
    event,
    path,
    start_slice,
    stop_slice,
    colormap='viridis'
):
    FREE_SPACE_IMPEDANCE_Z0 = 120*np.pi
    ELECTRON_VOLT = 1.6022e-19
    isr = event.raw_image_sensor_response
    power_north = isr.north**2/FREE_SPACE_IMPEDANCE_Z0
    power_west = isr.west**2/FREE_SPACE_IMPEDANCE_Z0
    energy_north = power_north*isr.time_slice_duration
    power_west = power_west*isr.time_slice_duration
    total_power = energy_north + power_west
    total_power_integral = np.sum(total_power[:, start_slice:stop_slice], axis=1)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.suptitle(simulation_truth_info_string(event.simulation_truth))
    ax = fig.add_axes([0.1 , 0.1, 0.8, 0.8])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('c_x / deg')
    ax.set_ylabel('c_y / deg')
    add2ax(
        ax=ax,
        pixel_amplitudes=total_power_integral/ELECTRON_VOLT,
        pixel_directions_x=np.rad2deg(
            event.image_sensor.pixel_directions[:, 0]),
        pixel_directions_y=np.rad2deg(
            event.image_sensor.pixel_directions[:, 1]),
        colormap=colormap,)
    fig.savefig(path)


def save_event_overview_images(
    event,
    output_directory,
    start_slice=None,
    stop_slice=None,
    colormap="viridis",
    scale = .7
):
    sensor_response = iaat.telescope.simulate_antenna_response(
        event.raw_image_sensor_response,
        antenna_efficiency=0.5,
        antenna_temperature=80,
        lower_frequency_cut=1.3e9,
        upper_frequency_cut=2.3e9,
        order=5)
    sr = sensor_response

    os.makedirs(output_directory, exist_ok=True)
    FREE_SPACE_IMPEDANCE_Z0 = 120*np.pi
    ELECTRON_VOLT = 1.6022e-19
    power_north = sr.north**2/FREE_SPACE_IMPEDANCE_Z0
    power_west = sr.west**2/FREE_SPACE_IMPEDANCE_Z0
    energy_north = power_north*sr.time_slice_duration
    energy_west = power_west*sr.time_slice_duration
    total_energy = energy_north + energy_west


    pixel_directions_x = np.rad2deg(event.image_sensor.pixel_directions[:, 0])
    pixel_directions_y = np.rad2deg(event.image_sensor.pixel_directions[:, 1])

    final_energy_integral = np.sum(
        total_energy[:, start_slice:stop_slice],
        axis=1)
    max_energy = np.max(final_energy_integral)

    image_idx = 0
    for time_slice in np.arange(start_slice, stop_slice):
        t = time_slice*event.simulation_truth['time_slice_duration']
        time_info = 't: ' + str(np.round(t*1e9, 1)) + 'ns'
        total_energy_integral = np.sum(
            total_energy[:, start_slice:time_slice], axis=1)

        north_abs_max = np.max(np.abs(sr.north[:, start_slice:stop_slice]))
        west_abs_max = np.max(np.abs(sr.west[:, start_slice:stop_slice]))
        comp_abs_max = np.max([north_abs_max, west_abs_max])

        fig = plt.figure(figsize=(16*scale, 9*scale), dpi=120/scale)
        fig.suptitle(
            simulation_truth_info_string(event.simulation_truth) + ', ' +
            time_info + '\n' + r"electric field/$\mu$Vm$^{-1}$" +
            "                                                         " +
            "energy/eV")

        ax_field_north = fig.add_axes([0. , 0.5, 0.4, 0.4])
        ax_field_north.spines['right'].set_visible(False)
        ax_field_north.spines['top'].set_visible(False)
        ax_field_north.set_xlabel('c_x / deg')
        ax_field_north.set_ylabel('c_y / deg')
        add2ax(
            ax=ax_field_north,
            pixel_amplitudes=1e6*sr.north[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            colormap='seismic',
            vmin=-1e6*comp_abs_max,
            vmax=1e6*comp_abs_max,)

        ax_field_west = fig.add_axes([0. , 0.1, 0.4, 0.4])
        ax_field_west.spines['right'].set_visible(False)
        ax_field_west.spines['top'].set_visible(False)
        ax_field_west.set_xlabel('c_x / deg')
        ax_field_west.set_ylabel('c_y / deg')
        add2ax(
            ax=ax_field_west,
            pixel_amplitudes=1e6*sr.west[:, time_slice],
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            colormap='seismic',
            vmin=-1e6*comp_abs_max,
            vmax=1e6*comp_abs_max,)

        ax_energy_integral = fig.add_axes([0.375 , 0.075, 0.6, 0.85])
        ax_energy_integral.spines['right'].set_visible(False)
        ax_energy_integral.spines['top'].set_visible(False)
        ax_energy_integral.set_xlabel('c_x / deg')
        ax_energy_integral.set_ylabel('c_y / deg')
        add2ax(
            ax=ax_energy_integral,
            pixel_amplitudes=total_energy_integral/ELECTRON_VOLT,
            pixel_directions_x=pixel_directions_x,
            pixel_directions_y=pixel_directions_y,
            colormap=colormap,
            vmin=0.,
            vmax=max_energy/ELECTRON_VOLT)
        image_idx += 1

        fig.savefig(
            os.path.join(
                output_directory,
                "{:06d}.jpg".format(image_idx)))
        plt.close(fig)


def save_event_overview_video(event_path, output_path):
    event = iaat.telescope.make_Event_from_tape_archive(event_path)
    with tempfile.TemporaryDirectory(prefix="askaryan_video_") as tmp_dir:
        save_event_overview_images(
            event=event,
            output_directory=tmp_dir,
            start_slice=120,
            stop_slice=180)
        make_video_from_image_slices(
            image_slice_dir=tmp_dir,
            out_path=output_path,
            image_slice_filename_wildcard='%06d.jpg')
