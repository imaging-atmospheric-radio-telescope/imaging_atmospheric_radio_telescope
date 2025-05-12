import argparse
import os
import sebastians_matplotlib_addons as sebplt
import imaging_atmospheric_askaryan_telescope as iaat
from imaging_atmospheric_askaryan_telescope import plot as iaat_plot
import numpy as np
import glob


parser = argparse.ArgumentParser(
    prog="central_feed_horn_scan.py",
    description=("Plot feed_horn_sub_scatter."),
)
parser.add_argument(
    "psf_dir",
    metavar="PSF_DIR",
    default="point spread function directory",
    type=str,
    help="Path to directory.",
)

sebplt.matplotlib.rcParams.update(iaat_plot.CONFIG["matplotlib"])
sebplt.matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

args = parser.parse_args()
psf_dir = args.psf_dir
out_dir = os.path.join(psf_dir, "plots", "central_feed_horn_scan")
os.makedirs(out_dir, exist_ok=True)

config = iaat.investigations.point_spread_function.utils.read_config(psf_dir)

source_key = "1"
for telescope_key in config["stars"]["telescopes"]:
    results = iaat.investigations.point_spread_function.stars.reduce_responses(
        work_dir=psf_dir,
        config=config,
        telescope_key=telescope_key,
        scenario_key="fully_inside_field_of_view",
    )
