import argparse
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_radio_telescope as iart
from imaging_atmospheric_radio_telescope import plot as iaat_plot
import optic_object_wavefronts as oow
import numpy as np
import spherical_coordinates
import merlict
import corsika_primary
import subprocess
import hashlib
import tempfile
import json_utils
import pandas
import scipy.spatial


parser = argparse.ArgumentParser(
    prog="compare_ray_tracing_optics.py",
    description=("Compare to ray tracing."),
)
parser.add_argument(
    "--out_dir",
    metavar="OUT_DIR",
    default="compare_ray_tracing_optics",
    type=str,
    help="Path to write figures to.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


args = parser.parse_args()
out_dir = args.out_dir

TELESCOPE_KEYS = ["crome", "medium_size_telescope", "large_size_telescope"]
MERLICT_PATH = "/home/relleums/Desktop/starter_kit/packages/merlict/merlict/c89/merlict_c89/build/bin/propagate_event_tape"

SENSOR_SCREEN_ID = 12


def estimate_region_of_interest(telescope):
    num_waves = 7
    nu_Hz = np.mean(
        iart.lownoiseblock.input_frequency_start_stop_Hz(telescope["lnb"])
    )
    wavelength_m = iart.signal.frequency_to_wavelength(nu_Hz)
    width_m = num_waves * wavelength_m
    theta_rad = np.arctan(width_m / telescope["mirror"]["focal_length_m"])
    num_bins = iart.investigations.point_spread_function.utils.substract_one_when_even(
        num_waves * 6
    )
    return theta_rad, num_bins


def make_telescope_scenery_for_ray_tracing(
    telescope, fn_polygon=71, fn_hex_grid=17
):
    scn = merlict.scenery.init()
    for key in [
        "perfect_mirror_reflection",
        "perfect_absorber_reflection",
        "vacuum_absorption",
        "vacuum_refraction",
    ]:
        scn["materials"]["spectra"][key] = (
            merlict.materials.spectra.init_from_resources(key)
        )

    scn["materials"]["media"]["vacuum"] = {
        "refraction_spectrum": "vacuum_refraction",
        "absorption_spectrum": "vacuum_absorption",
    }
    scn["materials"]["default_medium"] = "vacuum"
    scn["materials"]["surfaces"]["perfect_mirror"] = {
        "type": "cook-torrance",
        "reflection_spectrum": "perfect_mirror_reflection",
        "diffuse_weight": 0.0,
        "specular_weight": 1.0,
        "roughness": 0.0,
    }
    scn["materials"]["surfaces"]["perfect_absorber"] = {
        "type": "cook-torrance",
        "reflection_spectrum": "perfect_absorber_reflection",
        "diffuse_weight": 1.0,
        "specular_weight": 0.0,
        "roughness": 0.0,
    }

    scn["materials"]["boundary_layers"]["mirror_vacuum"] = {
        "inner": {"medium": "vacuum", "surface": "perfect_absorber"},
        "outer": {"medium": "vacuum", "surface": "perfect_mirror"},
    }
    scn["materials"]["boundary_layers"]["absorber_vacuum"] = {
        "inner": {"medium": "vacuum", "surface": "perfect_absorber"},
        "outer": {"medium": "vacuum", "surface": "perfect_absorber"},
    }

    mirror_z = 0.0
    mirror = oow.primitives.parabolic_cap_regular.init(
        outer_radius=telescope["mirror"]["outer_radius_m"],
        inner_radius=telescope["mirror"]["inner_radius_m"],
        focal_length=telescope["mirror"]["focal_length_m"],
        fn_polygon=fn_polygon,
        fn_hex_grid=fn_hex_grid,
        rot=0.0,
        ref="mirror",
    )
    mirror_frame = {
        "id": 3000,
        "pos": [0, 0, mirror_z],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "mirror",
        "mtl": {
            "mirror": "mirror_vacuum",
        },
    }

    screen_z = telescope["sensor"]["sensor_distance_m"]
    screen_outer_polygon = oow.geometry.regular_polygon.make_vertices_xy(
        # outer_radius=telescope["sensor"]["camera"]["outer_radius_m"],
        outer_radius=0.975 * telescope["mirror"]["inner_radius_m"],
        fn=fn_polygon,
        ref="screen_outer_bound",
        rot=0.0,
    )
    screen = oow.primitives.plane.init(
        outer_polygon=screen_outer_polygon,
        fn_hex_grid=fn_hex_grid,
        ref="screen",
    )
    screen_frame = {
        "id": SENSOR_SCREEN_ID,
        "pos": [0, 0, screen_z],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "screen",
        "mtl": {
            "screen": "absorber_vacuum",
        },
    }

    shield_z = screen_z * 1.01
    shield_outer_polygon = oow.geometry.regular_polygon.make_vertices_xy(
        outer_radius=telescope["mirror"]["inner_radius_m"],
        fn=fn_polygon,
        ref="shield_outer_bound",
        rot=0.0,
    )
    shield = oow.primitives.plane.init(
        outer_polygon=shield_outer_polygon,
        fn_hex_grid=fn_hex_grid,
        ref="shield",
    )
    shield_frame = {
        "id": 80,
        "pos": [0, 0, shield_z],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "shield",
        "mtl": {
            "shield": "absorber_vacuum",
        },
    }

    scn["geometry"]["objects"]["mirror"] = oow.export.reduce_mesh_to_obj(
        mirror
    )
    scn["geometry"]["relations"]["children"].append(mirror_frame)

    scn["geometry"]["objects"]["screen"] = oow.export.reduce_mesh_to_obj(
        screen
    )
    scn["geometry"]["relations"]["children"].append(screen_frame)

    scn["geometry"]["objects"]["shield"] = oow.export.reduce_mesh_to_obj(
        shield
    )
    scn["geometry"]["relations"]["children"].append(shield_frame)

    return scn


def seed_from_string(s):
    hhh = hashlib.md5(s.encode())
    return int(hhh.hexdigest(), 16)


def write_calibration_event(
    path, azimuth_rad, zenith_rad, radius_m, size, seed
):
    assert size >= 0
    prng = np.random.Generator(np.random.PCG64(seed))
    wavelength = 433e-9

    cx, cy = spherical_coordinates.az_zd_to_cx_cy(azimuth_rad, zenith_rad)

    RUNH = corsika_primary.I.RUNH
    EVTH = corsika_primary.I.EVTH

    BLOCK_SIZE = 100_000
    with corsika_primary.cherenkov.CherenkovEventTapeWriter(path) as run:
        runh = np.zeros(273, dtype=np.float32)
        runh[RUNH.MARKER] = RUNH.MARKER_FLOAT32
        runh[RUNH.RUN_NUMBER] = 1
        runh[RUNH.NUM_EVENTS] = 1
        run.write_runh(runh)

        evth = np.zeros(273, dtype=np.float32)
        evth[EVTH.MARKER] = EVTH.MARKER_FLOAT32
        evth[EVTH.RUN_NUMBER] = runh[RUNH.RUN_NUMBER]
        evth[EVTH.EVENT_NUMBER] = 1
        run.write_evth(evth)

        while size > 0:
            part = min([BLOCK_SIZE, size])
            size -= part

            bunches = corsika_primary.calibration_light_source.draw_parallel_and_isochor_bunches(
                cx=cx,
                cy=cy,
                aperture_radius=radius_m,
                wavelength=wavelength,
                size=part,
                prng=prng,
                speed_of_light=iart.signal.SPEED_OF_LIGHT_M_PER_S,
            )
            run.write_payload(bunches)


def estimate_light_intersection_on_screen(scenery_path, run_path, seed):
    call = [
        MERLICT_PATH,
        scenery_path,
        run_path,
        "0",
        "0",
        f"{seed:d}",
        f"{SENSOR_SCREEN_ID:d}",
    ]
    proc = subprocess.Popen(call, stdout=subprocess.PIPE)
    # proc.wait()

    x = []
    y = []
    for line in proc.stdout.readlines():
        sline = line.decode()
        sline = str.strip(sline)
        tokens = sline.split(",")
        x.append(float(tokens[2]))
        y.append(float(tokens[3]))
    xy = np.c_[x, y]
    return xy


def make_psf_image(
    telescope,
    telescope_feed_horn_tree,
    telescope_feed_horn_outer_radius,
    scenery_path,
    azimuth_rad,
    zenith_rad,
    size,
    seed,
):
    radius_thrown_m = 1.25 * telescope["mirror"]["outer_radius_m"]

    with tempfile.TemporaryDirectory(prefix="iart.") as tmp_dir:
        run_path = os.path.join(tmp_dir, f"{seed:03d}.tar")
        write_calibration_event(
            path=run_path,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            radius_m=radius_thrown_m,
            size=size,
            seed=seed,
        )

        xy = estimate_light_intersection_on_screen(
            scenery_path=scenery_path, run_path=run_path, seed=seed
        )

        roi_theta_rad, roi_num_bins = estimate_region_of_interest(telescope)
        telescope_roi = iart.investigations.point_spread_function.utils.make_telescope_like_other_but_with_region_of_interest_camera(
            source_azimuth_rad=azimuth_rad,
            source_zenith_rad=zenith_rad,
            region_of_interest_rad=roi_theta_rad,
            num_bins=roi_num_bins,
            other_telescope=telescope,
        )

        psf_image = {
            "x_bin_edges_m": telescope_roi["sensor"]["region_of_interest"][
                "x_bin_edges_m"
            ],
            "y_bin_edges_m": telescope_roi["sensor"]["region_of_interest"][
                "y_bin_edges_m"
            ],
        }
        psf_image["image"] = np.histogram2d(
            xy[:, 0],
            xy[:, 1],
            bins=(psf_image["x_bin_edges_m"], psf_image["y_bin_edges_m"]),
        )[0]
        area_thrown_m2 = np.pi * radius_thrown_m**2
        area_mirror_m2 = np.pi * (
            telescope["mirror"]["outer_radius_m"] ** 2
            - telescope["mirror"]["inner_radius_m"]
        )
        psf_image["si_unit"] = json_utils.dumps(
            {
                "radius_thrown_m": radius_thrown_m,
                "area_thrown_m2": area_thrown_m2,
                "area_mirror_m2": area_mirror_m2,
                "size_thrown": size,
                "size_in_mirror": size * area_mirror_m2 / area_thrown_m2,
            }
        )

        # feed horn assignment
        # --------------------
        num_feed_horns = telescope_feed_horn_tree.data.shape[0]
        feed_horn_index_bin_edges = np.linspace(
            -0.5, num_feed_horns - 0.5, num_feed_horns + 1
        )
        dd_m, ii = telescope_feed_horn_tree.query(xy)
        valid = dd_m <= telescope_feed_horn_outer_radius
        valid = valid.astype(float)
        feed_horn_image = np.histogram(
            ii, bins=feed_horn_index_bin_edges, weights=valid
        )[0]

    return psf_image, feed_horn_image


def report_add_roi_analysis(report, telescope, roi_analysis):
    report["roi_area_80p_m2"] = roi_analysis["area_quantile_m2"]
    report["roi_x_m"] = roi_analysis["argmax_x_m"]
    report["roi_y_m"] = roi_analysis["argmax_y_m"]
    az, zd = iart.utils.sky_and_screen.screen_x_y_to_sky_az_zd(
        x_m=report["roi_x_m"],
        y_m=report["roi_y_m"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    report["roi_azimuth_rad"] = az
    report["roi_zenith_rad"] = zd
    report["roi_solid_angle_80p_sr"] = (
        iart.utils.sky_and_screen.screen_area_to_sky_solid_angle(
            a_m2=report["roi_area_80p_m2"],
            focal_length_m=telescope["mirror"]["focal_length_m"],
        )
    )
    return report


NUM_EVENTS = 1000
SIZE = 10_000
PSF_QUANTILE = 0.8

for telescope_key in TELESCOPE_KEYS:
    tele_dir = os.path.join(out_dir, telescope_key)

    scenery_path = os.path.join(tele_dir, f"{telescope_key:s}.tar")

    # if not os.path.exists(scenery_path):
    iart.run.init(work_dir=tele_dir, telescope_key=telescope_key)
    telescope = iart.run.from_config(work_dir=tele_dir)["telescope"]
    sceneryPy = make_telescope_scenery_for_ray_tracing(telescope)
    merlict.scenery.write_tar(sceneryPy=sceneryPy, path=scenery_path)
    # sceneryPy = merlict.scenery.read_tar(path=scenery_path)

    fov = iart.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )
    telescope_feed_horn_tree = scipy.spatial.cKDTree(
        telescope["sensor"]["feed_horn_positions_m"][:, 0:2]
    )
    telescope_feed_horn_outer_radius = (
        iart.utils.hexagon_outer_radius_given_inner_radius(
            telescope["sensor"]["camera"]["feed_horn_inner_radius_m"]
        )
    )
    airy_radius_m = iart.telescope.calculate_airy_disk_radius_in_focal_plane(
        telescope=telescope
    )

    seeds_expected = set(np.arange(NUM_EVENTS))

    reports_path = os.path.join(tele_dir, "report.jsonl")
    if not os.path.exists(reports_path):
        json_utils.lines.write(reports_path, [])

    reports = json_utils.lines.read(reports_path)
    seeds_actual = set([r["seed"] for r in reports])

    seeds_missing = list(seeds_expected.difference(seeds_actual))

    for seed in seeds_missing:
        prng = np.random.Generator(np.random.PCG64(seed))

        azimuth_rad, zenith_rad = (
            spherical_coordinates.random.uniform_az_zd_in_cone(
                prng=prng,
                azimuth_rad=0.0,
                zenith_rad=0.0,
                min_half_angle_rad=0.0,
                max_half_angle_rad=fov[
                    "field_of_view_fully_outside_half_angle_rad"
                ],
            )
        )

        # event_path = os.path.join(tele_dir, f"{seed:06d}.tar")
        # if True:  # not os.path.exists(event_path):
        psf_image, feed_horn_image = make_psf_image(
            telescope=telescope,
            telescope_feed_horn_tree=telescope_feed_horn_tree,
            telescope_feed_horn_outer_radius=telescope_feed_horn_outer_radius,
            scenery_path=scenery_path,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            size=SIZE,
            seed=seed,
        )
        # iart.calibration.save(path=event_path, psf_image=psf_image)
        # psf_image = iart.calibration.load(path=event_path)

        report = {}
        report["seed"] = seed
        report["source_azimuth_rad"] = azimuth_rad
        report["source_zenith_rad"] = zenith_rad
        report["source_expected_energy_J"] = json_utils.loads(
            psf_image["si_unit"]
        )["size_in_mirror"]

        roi_analysis = iart.investigations.point_spread_function.power_image_analysis.analyse_image(
            x_bin_edges_m=psf_image["x_bin_edges_m"],
            y_bin_edges_m=psf_image["y_bin_edges_m"],
            image=psf_image["image"],
            containment_quantile=PSF_QUANTILE,
        )
        report = report_add_roi_analysis(
            report=report, telescope=telescope, roi_analysis=roi_analysis
        )
        report["roi_r80_m"] = (
            iart.investigations.point_spread_function.power_image_analysis.encircle_containment(
                x_bin_edges_m=psf_image["x_bin_edges_m"],
                y_bin_edges_m=psf_image["y_bin_edges_m"],
                image=psf_image["image"],
                x_m=roi_analysis["argmax_x_m"],
                y_m=roi_analysis["argmax_y_m"],
                quantile=PSF_QUANTILE,
            )
        )

        feed_horns_signal_mask = iart.investigations.point_spread_function.utils.make_feed_horns_signal_mask(
            feed_horn_positions_m=telescope["sensor"]["feed_horn_positions_m"],
            x_m=roi_analysis["argmax_x_m"],
            y_m=roi_analysis["argmax_y_m"],
            r_m=2 * airy_radius_m,
        )
        feed_horns_background_mask = np.logical_not(feed_horns_signal_mask)

        energy_signal_J = feed_horn_image[feed_horns_signal_mask]
        energy_background_J = feed_horn_image[feed_horns_background_mask]

        total_energy_signal_J = np.sum(energy_signal_J)
        mean_energy_signal_J = np.mean(energy_signal_J)
        median_energy_background_J = np.percentile(energy_background_J, 50)

        signal_to_noise_ratio = (
            mean_energy_signal_J / median_energy_background_J
        )
        energy_conservation_ratio = (
            total_energy_signal_J / report["source_expected_energy_J"]
        )

        report["feed_horn_energy_conservation_ratio"] = (
            energy_conservation_ratio
        )
        report["feed_horn_total_energy_signal_J"] = total_energy_signal_J
        report["feed_horn_median_energy_background_J"] = (
            median_energy_background_J
        )
        report["feed_horn_signal_to_noise_ratio"] = signal_to_noise_ratio

        reports.append(report)
    json_utils.lines.write(reports_path, reports)


for telescope_key in TELESCOPE_KEYS:
    tele_dir = os.path.join(out_dir, telescope_key)

    telescope = iart.run.from_config(work_dir=tele_dir)["telescope"]
    fov = iart.investigations.point_spread_function.utils.make_field_of_view_region_edges(
        sensor=telescope["sensor"],
        focal_length_m=telescope["mirror"]["focal_length_m"],
    )

    reports_path = os.path.join(tele_dir, "report.jsonl")
    snap = iart.investigations.point_spread_function.utils.read_jsonl_reports_into_recarray(
        reports_path
    )

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        np.pi * snap["roi_r80_m"] ** 2,
        color="black",
        alpha=0.1,
        linewidth=0.0,
        marker="o",
    )
    fig.savefig(os.path.join(tele_dir, f"spread_simple.jpg"))
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        np.rad2deg(snap["roi_zenith_rad"])
        / np.rad2deg(snap["source_zenith_rad"]),
        color="black",
        alpha=0.1,
        linewidth=0.0,
        marker="o",
    )
    fig.savefig(os.path.join(tele_dir, f"distortion_simple.jpg"))
    sebplt.close(fig)

    fig = sebplt.figure(style={"rows": 1080, "cols": 1920, "fontsize": 2.0})
    ax = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    ax.plot(
        np.rad2deg(snap["source_zenith_rad"]),
        snap["feed_horn_energy_conservation_ratio"],
        color="black",
        alpha=0.1,
        linewidth=0.0,
        marker="o",
    )
    fig.savefig(os.path.join(tele_dir, f"energy_conservation_simple.jpg"))
    sebplt.close(fig)

    # Analysis

    psf_off_deg = np.rad2deg(snap["source_zenith_rad"])

    oa_bin = (
        iart.investigations.point_spread_function.utils.guess_off_axis_binning(
            num_samples=len(psf_off_deg),
            half_angle=np.rad2deg(fov["field_of_view_half_angle_rad"]),
        )
    )

    psf_area_m2 = np.pi * snap["roi_r80_m"] ** 2
    h_psf_area = (
        iart.investigations.point_spread_function.utils.histogram_p50_s68(
            x=psf_off_deg, y=psf_area_m2, edges=oa_bin["edges"]
        )
    )
    psf_fit, psf_fit_std = (
        iart.investigations.point_spread_function.utils.fit_poly1d(
            x=oa_bin["centers"],
            y=h_psf_area["p50"],
        )
    )

    disto = snap["roi_zenith_rad"] / snap["source_zenith_rad"]
    h_disto = (
        iart.investigations.point_spread_function.utils.histogram_p50_s68(
            x=psf_off_deg, y=disto, edges=oa_bin["edges"]
        )
    )
    distortion_fit, distortion_fit_std = (
        iart.investigations.point_spread_function.utils.fit_poly1d(
            x=oa_bin["centers"],
            y=h_disto["p50"],
        )
    )

    h_enecon = (
        iart.investigations.point_spread_function.utils.histogram_p50_s68(
            x=psf_off_deg,
            y=snap["feed_horn_energy_conservation_ratio"],
            edges=oa_bin["edges"],
        )
    )
    eneFit, eneFit_std = (
        iart.investigations.point_spread_function.utils.fit_poly1d(
            x=oa_bin["centers"],
            y=h_enecon["p50"],
        )
    )

    summary = {
        "off_axis_bin_deg": oa_bin,
        "energy_conservation_1": {
            "hist": h_enecon,
            "fit": eneFit,
            "fit_std": eneFit_std,
            "fit_method": "linear-fit-y-axis-intersection",
        },
        "point_spread_function_m2": {
            "hist": h_psf_area,
            "fit": psf_fit,
            "fit_std": psf_fit_std,
        },
        "distortion_1": {
            "hist": h_disto,
            "fit": distortion_fit,
            "fit_std": distortion_fit_std,
        },
    }
    with open(
        os.path.join(out_dir, f"{telescope_key:s}.summary.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(summary, indent=4))
