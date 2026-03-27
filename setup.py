import setuptools
import os

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(
    os.path.join("imaging_atmospheric_askaryan_telescope", "version.py")
) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="imaging_atmospheric_askaryan_telescope",
    version=version,
    description=(
        "Simulate and investigate the "
        "Imaging Atmospheric Askarian Telescope for gamma ray astronomy."
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/relleums/imaging_atmospheric_askaryan_telescope",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "imaging_atmospheric_askaryan_telescope",
        "imaging_atmospheric_askaryan_telescope.utils",
        "imaging_atmospheric_askaryan_telescope.corsika",
        "imaging_atmospheric_askaryan_telescope.corsika.build",
        "imaging_atmospheric_askaryan_telescope.corsika.coreas",
        "imaging_atmospheric_askaryan_telescope.calibration_source",
        "imaging_atmospheric_askaryan_telescope.production",
        "imaging_atmospheric_askaryan_telescope.telescopes",
        "imaging_atmospheric_askaryan_telescope.investigations",
        "imaging_atmospheric_askaryan_telescope.investigations.point_spread_function",
        "imaging_atmospheric_askaryan_telescope.investigations.airshower_response",
        "imaging_atmospheric_askaryan_telescope.sites",
    ],
    package_data={
        "imaging_atmospheric_askaryan_telescope": [
            os.path.join("corsika", "build", "resources", "config.h"),
            os.path.join("telescopes", "resources", "*.json"),
            os.path.join("sites", "resources", "*.json"),
            os.path.join(
                "investigations", "point_spread_function", "scripts", "*.py"
            ),
        ]
    },
    install_requires=[
        "sebastians_matplotlib_addons>=0.0.19",
        "json_utils_sebastian-achim-mueller>=0.0.5",
        "rename_after_writing>=0.0.12",
        "homogeneous_transformation>=0.0.1.1.7.2",
        "spherical_coordinates>=0.1.8",
        "binning_utils_sebastian-achim-mueller>=0.0.20",
        "solid_angle_utils>=0.1.2",
        "thin_lens>=0.0.3",
        "optic_object_wavefronts>=1.3.17",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
)
