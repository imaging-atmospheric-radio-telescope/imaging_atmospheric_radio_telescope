######################################
Imaging Atmospheric Askaryan Telescope
######################################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

Exploring the detection of cosmic gamma rays from ground by imaging the radio
emission of air showers.


|example_event|


The detection of cosmic gamma-rays with the atmospheric Cherenkov-method is
limited to ~1000 hours of observation time per year. An atmospheric method based
on the radio emission of air showers might nine-fold this time because it can
observe during the day. Here I investigate an imaging radio telescope that
senses the radio emission of air showers. Its images look similar to those of
Cherenkov telescopes. I simulate the optics, not using ray tracing, but
wave mechanics. Such an 'Askaryan telescope' can adopt electronics from
Cherenkov telescopes. Its mirrors can be made out of wire mesh. The goal of this
package is to explore the capabilities and practicality of such an
Askaryan telescope.

*******
Install
*******

The python package.

.. code-block:: bash

    pip install --editable imaging_atmospheric_askaryan_telescope/


Enter the python package directory.

.. code-block:: bash

    cd imaging_atmospheric_askaryan_telescope


Download ``CORSIKA-CoREAS`` from KIT.
You need to ask the developers of CORSIKA for the username and password.

.. code-block:: python

    import imaging_atmospheric_askaryan_telescope as iaat

    iaat.corsika.coreas.install.download(
        output_dir="build",
        username="XXX",
        password="YYY",
    )


Check if it is the exact version we expect.

.. code-block:: python

    iaat.corsika.coreas.install.is_expected_version(
        corsika_tar_gz_path="build/corsika-77100.tar.gz",
    )

Finally build CORSIKA-CoREAS in the 'build' directory.

.. code-block:: python

    iaat.corsika.coreas.install.build(
        corsika_tar_gz_path="build/corsika-77100.tar.gz",
        build_dir="build",
    )


****
Test
****

This is not a unit-test. It is more like a minimal example.

.. code-block:: bash

    python imaging_atmospheric_askaryan_telescope/scripts/test.py



.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |TestStatus| image:: https://github.com/relleums/imaging_atmospheric_askaryan_telescope/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/relleums/imaging_atmospheric_askaryan_telescope/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/imaging_atmospheric_askaryan_telescope
    :target: https://pypi.org/project/imaging_atmospheric_askaryan_telescope

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |MITLicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. |example_event| image:: https://github.com/relleums/imaging_atmospheric_askaryan_telescope/blob/main/readme/example_event.gif
