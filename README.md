# Imaging Atmospheric Askaryan Telescope
![img](readme/example_event.gif)
Exploring the detection of cosmic gamma-rays from ground using the radio-emission of showers.

## Abstract
The detection of cosmic gamma-rays with the atmospheric Cherenkov-method is limited to ~1000h of observation-time per year.
An atmospheric method based on the radio-emission of showers might nine-fold this time because it can observe during the day.
Here I investigate an imaging-radio-telescope that senses the radio-emission of showers.
Its images look similar to those of Cherenkov-telescopes.
I simulate the optics, not using ray-tracing, but wave-mechanics.
Such an "Askaryan-telescope" can adopt electronics from Cherenkov-telescopes.
Its mirrors can be made out of wire-mesh.
The goal of this package is to explore the capabilities and practicality of such an Askaryan-telescope.

## Install

The python-package.

```bash
pip install -e imaging_atmospheric_askaryan_telescope/
```

Enter the python-package directory.
```bash
cd imaging_atmospheric_askaryan_telescope
```

Download CORSIKA-CoREAS from KIT.
You need to ask the developers of CORSIKA for the username and password.

```python
import imaging_atmospheric_askaryan_telescope as iaat

iaat.corsika.coreas.install.download(
    output_dir="build",
    username="XXX",
    password="YYY",
)
```

Check if it is the exact version we expect.


```python
iaat.corsika.coreas.install.is_expected_version(
    corsika_tar_gz_path="build/corsika-77100.tar.gz",
)
```

Finally build CORSIKA-CoREAS in the 'build' directory.

```python
iaat.corsika.coreas.install.build(
    corsika_tar_gz_path="build/corsika-77100.tar.gz",
    build_dir="build",
)
```

## test
This is not a unit-test. It is more like a minimal example.
```bash
python imaging_atmospheric_askaryan_telescope/scripts/test.py
```
