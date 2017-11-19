from distutils.core import setup

setup(
    name='askarian_telescope',
    version='0.0.0',
    description='Simulate and investigate an Imaging Air-shower Askarian Telescope',
    url='https://github.com/fact-project/',
    author='Sebastian Achim Mueller',
    author_email='sebmuell@phys.ethz.ch',
    license='GPLv3',
    packages=[
        'askarian_telescope',
    ],
    package_data={
        'askarian_telescope': []
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
