# Copyright 2017 Sebastian A. Mueller
from distutils.core import setup

setup(
    name='imaging_atmospheric_askaryan_telescope',
    version='0.0.1',
    description='Simulate and investigate the ' +
    'Imaging Atmospheric Askarian Telescope for gamma-ray astronomy.',
    url='https://github.com/relleums/imaging_atmospheric_askaryan_telescope',
    author='Sebastian Achim Mueller',
    author_email='sebmuell@phys.ethz.ch',
    license='GPLv3',
    packages=[
        'imaging_atmospheric_askaryan_telescope',
    ],
    package_data={
        'imaging_atmospheric_askaryan_telescope': []
    },
    install_requires=[
        'docopt',
        'scipy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': []
    },
    zip_safe=False,
)
