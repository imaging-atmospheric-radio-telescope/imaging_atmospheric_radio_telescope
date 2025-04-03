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
        "imaging_atmospheric_askaryan_telescope.corsika.coreas",
        "imaging_atmospheric_askaryan_telescope.corsika.coreas.install",
        "imaging_atmospheric_askaryan_telescope.production",
    ],
    package_data={
        "imaging_atmospheric_askaryan_telescope": [
            os.path.join(
                "corsika", "coreas", "install", "resources", "config.h"
            )
        ]
    },
    install_requires=[
        #"sebastians_matplotlib_addons>=0.0.17",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
)
