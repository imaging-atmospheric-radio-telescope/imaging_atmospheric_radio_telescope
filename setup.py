# Copyright 2017 Sebastian A. Mueller
import setuptools
import os

setuptools.setup(
    name="imaging_atmospheric_askaryan_telescope",
    version="0.0.4",
    description="Simulate and investigate the "
    + "Imaging Atmospheric Askarian Telescope for gamma-ray astronomy.",
    url="https://github.com/relleums/imaging_atmospheric_askaryan_telescope",
    author="Sebastian Achim Mueller",
    author_email="sebmuell@phys.ethz.ch",
    license="GPLv3",
    packages=["imaging_atmospheric_askaryan_telescope",],
    package_data={
        "imaging_atmospheric_askaryan_telescope": [
            os.path.join(
                "corsika", "coreas", "install", "resources", "config.h"
            )
        ]
    },
    install_requires=["docopt", "scipy", "matplotlib",],
    entry_points={"console_scripts": []},
    zip_safe=False,
)
