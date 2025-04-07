######################################
Imaging Atmospheric Askaryan Telescope
######################################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

Exploring the detection of cosmic gamma rays from ground by imaging the radio
emission of air showers.


|example_event|


The detection of cosmic gamma rays with the atmospheric Cherenkov method is
limited to about 1000 hours of dark and clear observation time per year. An
atmospheric method based on the radio emission of air showers might nine-fold
this time because it can observe during the bright day. Here we investigate an
imaging radio telescope that senses the radio emission of air showers to record
pictures. First simulations indicate that its images look similar to those of
Cherenkov telescopes. We simulate the optics, not using ray tracing, but wave
mechanics. Such an 'Askaryan telescope' can adopt electronics from Cherenkov
telescopes. Its mirrors can be made out of wire mesh. The goal of this package
is to estimate the response function of such an Askaryan telescope to cosmic
rays and cosmic gamma rays.

*******
Install
*******

Clone the python package from GitHub.

.. code-block:: bash

    git clone git@github.com:relleums/imaging_atmospheric_askaryan_telescope.git


and install it using ``pip``.

.. code-block:: bash

    pip install --editable imaging_atmospheric_askaryan_telescope/


Enter the python package directory.

.. code-block:: bash

    cd imaging_atmospheric_askaryan_telescope

Download ``corsika-77100.tar.gz`` from KIT and save it in:

.. code-block:: bash

    imaging_atmospheric_askaryan_telescope/corsika/__build__/corsika-77100.tar.gz


You need to ask the developers of ``CORSIKA`` for the ``username`` and
``password``.


Build ``CORSIKA-CoREAS`` for the ``imaging_atmospheric_askaryan_telescope`` by
calling:

.. code-block:: python
    import imaging_atmospheric_askaryan_telescope

    imaging_atmospheric_askaryan_telescope.corsika.build.install()


If all works out as expected, you should see:

.. code-block::

    Building CORSIKA CoREAS ... SUCCESS.


*********
Uninstall
*********

First uninstall the ``CORSIKA`` build.

.. code-block:: python
    import imaging_atmospheric_askaryan_telescope

    imaging_atmospheric_askaryan_telescope.corsika.build.uninstall()


Finally, uninstall the python package.

.. code-block:: bash

    pip uninstall imaging_atmospheric_askaryan_telescope


****
Test
****

.. code-block:: bash

    pytest imaging_atmospheric_askaryan_telescope/


For a minimal example run:

.. code-block:: bash

    python imaging_atmospheric_askaryan_telescope/scripts/test.py


**********
References
**********

From Sebastian's library (``bib search radio, lnb, antenna``).

.. code-block::

    balanis2015antenna : balanis2015antenna.pdf
    ------------------
        Antenna theory: {A}nalysis and design
        Balanis, Constantine A
        2015

    alvarez2013midas : alvarez2013midas.pdf
    ----------------
        The {MIDAS} telescope for microwave detection of ultra-high energy cosmic rays
        Alvarez-Muniz, J and Soares, E Amaral and Berlin, A and Bogdan, M and
        2013

    huege2016radio : huege2016radio.pdf
    --------------
        Radio detection of cosmic ray air showers in the digital era
        Huege, Tim
        2016

    falcke2005detection : falcke2005detection.pdf
    -------------------
        Detection and imaging of atmospheric radio flashes from cosmic ray air showers
        Falcke, H and Apel, WD and Badea, AF and B{\"a}hren, L and Bekk, K and Bercuci,
        2005

    gordon1964arecibo : gordon1964arecibo.pdf
    -----------------
        Arecibo ionospheric observatory
        Gordon, William E
        1964

    blumer2008air : blumer2008air.pdf
    -------------
        Air shower radio detection with {LOPES}
        Bl{\"u}mer, J and Apel, WD and Arteaga, JC and Asch, T and Auffenberg, J and
        2008

    huege2013simulating : huege2013simulating.pdf
    -------------------
        Simulating radio emission from air showers with {CoREAS}
        Huege, Tim and Ludwig, Marianne and James, Clancy W
        2013

    ludwig2011reas3 : ludwig2011reas3.pdf
    ---------------
        {REAS3}: {Monte} {Carlo} simulations of radio emission from cosmic ray air
        Ludwig, Marianne and Huege, Tim
        2011

    huege2004diss : huege2004diss.pdf
    -------------
        Radio Emission from Cosmic Ray Air Showers
        Huege, Tim
        2004

    schroder2012noise : schroder2012noise.pdf
    -----------------
        On noise treatment in radio measurements of cosmic ray air showers
        Schr{\"o}der, FG and Apel, WD and Arteaga, JC and Asch, T and B{\"a}hren, L and
        2012

    werner2013phd : werner2013phd.pdf
    -------------
        Detection of Microwave Emission of Extensive Air Showers with the {CROME}
        Werner, Felix
        2013

    smida2011first : smida2011first.pdf
    --------------
        First results of the {CROME} experiment
        Smida, Radomir and Bluemer, H and Engel, R and Haungs, A and Huege, T and
        2011


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
