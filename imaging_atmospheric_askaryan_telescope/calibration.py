from . import production
from . import calibration_source
from . import lownoiseblock
from . import investigations
from . import utils
from . import timing_and_sampling

import tempfile
import numpy as np
import io
import os
import json_utils
import rename_after_writing as rnw


def make_calibration(telescope, time_oversampling=6):
    _timing = timing_and_sampling.make_timing_from_lnb(
        lnb=telescope["lnb"],
        oversampling=time_oversampling,
        readout_sampling_rate_per_s=1e9,
        time_window_duration_s=10e-9,
    )
    calib = {}
    calib["image"] = make_point_spread_function_image(
        telescope=telescope,
        timing=_timing,
    )
    calib["containment"] = analyse_point_spread_function_image(
        psf_image=calib["image"],
    )

    # feed horn
    containment_watershed = interpolate(
        x=telescope["sensor"]["feed_horn_area_m2"],
        xp=calib["containment"]["area_quantile_watershed_m2"],
        fp=calib["containment"]["quantiles"],
        fmax=1.0,
    )
    containment_encirclement = interpolate(
        x=telescope["sensor"]["feed_horn_area_m2"],
        xp=calib["containment"]["area_quantile_encirclement_m2"],
        fp=calib["containment"]["quantiles"],
        fmax=1.0,
    )
    calib["point_spread_function_quantile_contained_in_feed_horn"] = {
        "watershed": containment_watershed,
        "encirclement": containment_encirclement,
    }

    # feed horn scatter center
    containment_scatter_watershed = interpolate(
        x=telescope["sensor"]["feed_horn_scatter_center_area_m2"],
        xp=calib["containment"]["area_quantile_watershed_m2"],
        fp=calib["containment"]["quantiles"],
        fmax=1.0,
    )
    containment_scatter_encirclement = interpolate(
        x=telescope["sensor"]["feed_horn_scatter_center_area_m2"],
        xp=calib["containment"]["area_quantile_encirclement_m2"],
        fp=calib["containment"]["quantiles"],
        fmax=1.0,
    )
    calib[
        "point_spread_function_quantile_contained_in_feed_horn_scatter_center"
    ] = {
        "watershed": containment_scatter_watershed,
        "encirclement": containment_scatter_encirclement,
    }

    return calib


def interpolate(x, xp, fp, fmax):
    f = np.interp(x=x, xp=xp, fp=fp)
    f = fmax if f > fmax else f
    return f


def add_calibration_to_telescope(telescope, path):
    if not os.path.exists(path):
        calibration = make_calibration(telescope=telescope)
        save_calibration(path=path, calibration=calibration)

    telescope["calibration"] = load_calibration(path=path)

    return telescope


def make_point_spread_function_image(
    telescope,
    timing,
    work_dir=None,
    region_of_interest_num_bins=61,
    logger=None,
):
    onaxis_source_config = make_onaxis_source_config(telescope=telescope)
    region_of_interest_rad = guess_region_of_interest_full_angle(
        telescope=telescope
    )

    SITE_DOES_NOT_MATTER = make_site()
    telescope_region_of_interest = investigations.point_spread_function.utils.make_telescope_like_other_but_with_region_of_interest_camera(
        source_azimuth_rad=0.0,
        source_zenith_rad=0.0,
        region_of_interest_rad=region_of_interest_rad,
        num_bins=region_of_interest_num_bins,
        other_telescope=telescope,
    )

    if work_dir is None:
        work_dir_handle = tempfile.TemporaryDirectory(prefix="iaat-")
        work_dir = work_dir_handle.name
    else:
        work_dir_handle = None

    NON_RELEVANT_RANDOM_SEED = 1

    production.simulate_telescope_response(
        out_dir=work_dir,
        source_config=onaxis_source_config,
        site=SITE_DOES_NOT_MATTER,
        telescope=telescope_region_of_interest,
        timing=timing,
        thermal_noise_random_seed=NON_RELEVANT_RANDOM_SEED + 1,
        readout_random_seed=NON_RELEVANT_RANDOM_SEED + 2,
        camera_lnb_random_seed=NON_RELEVANT_RANDOM_SEED + 3,
        stop_after_section="feed_horns",
        save_feed_horns_electric_fields=True,
        logger=logger,
    )

    x_bin_edges_m, y_bin_edges_m = (
        _make_bin_edges_from_region_of_interest_sensor(
            sensor=telescope_region_of_interest["sensor"]
        )
    )
    image = _load_image_from_production_response(
        production_work_dir=work_dir,
        region_of_interest_num_bins=region_of_interest_num_bins,
    )
    image /= image.sum()

    if work_dir_handle is not None:
        work_dir_handle.cleanup()

    psf_image = {
        "x_bin_edges_m": x_bin_edges_m,
        "y_bin_edges_m": y_bin_edges_m,
        "image": image,
        "si_unit": "1",
    }
    return psf_image


def analyse_point_spread_function_image(psf_image, quantiles=None):

    if quantiles is None:
        quantiles = np.linspace(0.01, 0.99, 99)

    area_wts_quantiles_m2 = []
    area_enc_quantiles_m2 = []
    for quantile in quantiles:
        _ana = investigations.point_spread_function.power_image_analysis.analyse_image(
            x_bin_edges_m=psf_image["x_bin_edges_m"],
            y_bin_edges_m=psf_image["y_bin_edges_m"],
            image=psf_image["image"],
            containment_quantile=quantile,
        )
        area_wts_quantiles_m2.append(_ana["area_quantile_m2"])

        r_quantile_m = investigations.point_spread_function.power_image_analysis.encircle_containment(
            x_bin_edges_m=psf_image["x_bin_edges_m"],
            y_bin_edges_m=psf_image["y_bin_edges_m"],
            image=psf_image["image"],
            x_m=0.0,
            y_m=0.0,
            quantile=quantile,
        )
        area_enc_quantiles_m2.append(np.pi * r_quantile_m**2)

    area_wts_quantiles_m2 = np.array(area_wts_quantiles_m2)
    area_enc_quantiles_m2 = np.array(area_enc_quantiles_m2)

    return {
        "quantiles": quantiles,
        "area_quantile_watershed_m2": area_wts_quantiles_m2,
        "area_quantile_encirclement_m2": area_enc_quantiles_m2,
    }


def make_onaxis_source_config(telescope):
    lnb_start_Hz, lnb_stop_Hz = lownoiseblock.input_frequency_start_stop_Hz(
        lnb=telescope["lnb"]
    )
    lnb_input_frequency_Hz = np.mean([lnb_start_Hz, lnb_stop_Hz])

    ARBITRARY_POWER_W = 1
    ARBITRARY_DISTANCE_M = 1e3
    onaxis_source_config = production.radio_from_plane_wave.make_config()
    s1 = calibration_source.plane_wave_in_far_field.make_config()
    s1["geometry"]["azimuth_rad"] = 0.0
    s1["geometry"]["zenith_rad"] = 0.0

    s1["power"][
        "power_of_isotrop_and_point_like_emitter_W"
    ] = ARBITRARY_POWER_W
    s1["power"][
        "distance_to_isotrop_and_point_like_emitter_m"
    ] = ARBITRARY_DISTANCE_M

    s1["sine_wave"]["emission_frequency_Hz"] = lnb_input_frequency_Hz
    s1["sine_wave"]["emission_duration_s"] = 5e-9
    s1["sine_wave"]["emission_ramp_up_duration_s"] = 1e-9
    s1["sine_wave"]["emission_ramp_down_duration_s"] = 1e-9
    s1["sine_wave"]["emission_overhead_duration_before_and_after_s"] = 1e-9

    onaxis_source_config["plane_waves"] = {}
    onaxis_source_config["plane_waves"]["onaxis"] = s1

    return onaxis_source_config


def guess_region_of_interest_full_angle(telescope):
    D = 2.0 * telescope["mirror"]["outer_radius_m"]
    f = telescope["mirror"]["focal_length_m"]
    overhead = 6.0
    return (D / f) * overhead * telescope["airy_half_angle_rad"]


def make_site():
    return {
        "observation_level_asl_m": 0,
        "earth_magnetic_field_x_muT": 0,
        "earth_magnetic_field_z_muT": 0,
        "name": "Arbitrary",
    }


def save_calibration(path, calibration):
    os.makedirs(path, exist_ok=True)
    save(path=os.path.join(path, "image.tar"), psf_image=calibration["image"])
    with rnw.open(os.path.join(path, "containment.json"), "wt") as f:
        f.write(json_utils.dumps(calibration["containment"], indent=4))
    fn = "point_spread_function_quantile_contained_in_feed_horn"
    with rnw.open(os.path.join(path, f"{fn:s}.json"), "wt") as f:
        f.write(json_utils.dumps(calibration[fn], indent=4))
    fs = "point_spread_function_quantile_contained_in_feed_horn_scatter_center"
    with rnw.open(os.path.join(path, f"{fs:s}.json"), "wt") as f:
        f.write(json_utils.dumps(calibration[fs], indent=4))


def load_calibration(path):
    calib = {}
    calib["image"] = load(path=os.path.join(path, "image.tar"))
    with open(os.path.join(path, "containment.json"), "rt") as f:
        calib["containment"] = json_utils.loads(f.read())
    fn = "point_spread_function_quantile_contained_in_feed_horn"
    with open(os.path.join(path, f"{fn:s}.json"), "rt") as f:
        calib[fn] = json_utils.loads(f.read())
    fs = "point_spread_function_quantile_contained_in_feed_horn_scatter_center"
    with open(os.path.join(path, f"{fs:s}.json"), "rt") as f:
        calib[fs] = json_utils.loads(f.read())
    return calib


def save(path, psf_image):
    with utils.tarstream.TarStream(path=path, mode="w") as t:
        t.write(
            filename="x_bin_edges_m.npy",
            filebytes=npy_to_bytes(psf_image["x_bin_edges_m"]),
        )
        t.write(
            filename="y_bin_edges_m.npy",
            filebytes=npy_to_bytes(psf_image["y_bin_edges_m"]),
        )
        t.write(
            filename="image.npy",
            filebytes=npy_to_bytes(psf_image["image"]),
        )
        t.write(
            filename="si_unit.txt",
            filebytes=npy_to_bytes(psf_image["si_unit"]),
        )


def load(path):
    out = {}
    with utils.tarstream.TarStream(path=path, mode="r") as t:
        filename, filebytes = t.read()
        assert filename == "x_bin_edges_m.npy"
        out["x_bin_edges_m"] = bytes_to_npy(filebytes)

        filename, filebytes = t.read()
        assert filename == "y_bin_edges_m.npy"
        out["y_bin_edges_m"] = bytes_to_npy(filebytes)

        filename, filebytes = t.read()
        assert filename == "image.npy"
        out["image"] = bytes_to_npy(filebytes)

        filename, filebytes = t.read()
        assert filename == "si_unit.txt"
        out["si_unit"] = bytes_to_npy(filebytes)

    return out


def _load_image_from_production_response(
    production_work_dir, region_of_interest_num_bins
):
    _path = os.path.join(
        production_work_dir,
        "feed_horns",
        "energy.npy",
    )
    with open(_path, "rb") as f:
        image = np.load(f)

    image = image.reshape(
        (region_of_interest_num_bins, region_of_interest_num_bins)
    )
    return image


def _make_bin_edges_from_region_of_interest_sensor(sensor):
    return (
        sensor["region_of_interest"]["x_bin_edges_m"],
        sensor["region_of_interest"]["y_bin_edges_m"],
    )


def npy_to_bytes(a):
    tmp = io.BytesIO()
    np.save(tmp, a)
    tmp.seek(0)
    return tmp.read()


def bytes_to_npy(b):
    tmp = io.BytesIO()
    tmp.write(b)
    tmp.seek(0)
    return np.load(tmp)
