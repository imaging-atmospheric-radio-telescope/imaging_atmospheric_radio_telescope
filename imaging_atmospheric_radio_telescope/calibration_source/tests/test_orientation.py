from imaging_atmospheric_radio_telescope.calibration_source.plane_wave_in_far_field import (
    compile_homogeneous_transformation,
)
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.linalg import norm
from numpy import pi as PI
from numpy import cos
from numpy import sin
from numpy import sqrt

import numpy as np
import homogeneous_transformation as homtra


VX = np.array([1, 0, 0])
VY = np.array([0, 1, 0])
VZ = np.array([0, 0, 1])

V_NULL = np.array([0, 0, 0])
V_E_FIELD = VX
V_B_FIELD = VY
V_PROPAGATION = VZ

ISQRT2 = 1.0 / sqrt(2)


def test_zero():
    t = compile_homogeneous_transformation(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        polarization_angle_rad=0.0,
    )

    x = homtra.transform_orientation(t, VX)
    assert_array_almost_equal(actual=x, desired=VX)

    y = homtra.transform_orientation(t, VY)
    assert_array_almost_equal(actual=y, desired=VY)

    z = homtra.transform_orientation(t, VZ)
    assert_array_almost_equal(actual=z, desired=VZ)


def rot_x(v, angle):
    x, y, z = v
    return np.array(
        [
            x,
            cos(angle) * y - sin(angle) * z,
            sin(angle) * y - cos(angle) * z,
        ]
    )


def rot_y(v, angle):
    x, y, z = v
    return np.array(
        [
            cos(angle) * x + sin(angle) * z,
            y,
            -sin(angle) * x + cos(angle) * z,
        ]
    )


def rot_z(v, angle):
    x, y, z = v
    return np.array(
        [
            cos(angle) * x - sin(angle) * y,
            sin(angle) * x + cos(angle) * y,
            z,
        ]
    )


def assert_transform_orientation_has_no_effect(t, v):
    a = homtra.transform_orientation(t=t, d=v)
    assert_array_almost_equal(actual=a, desired=v)


def assert_no_translation(t):
    pos = homtra.get_translation_vector(t=t)
    assert_array_almost_equal(actual=pos, desired=V_NULL)


def test_rot_z():
    x = rot_z(v=VX, angle=0.0)
    assert_array_almost_equal(actual=x, desired=VX)
    y = rot_z(v=VY, angle=0.0)
    assert_array_almost_equal(actual=y, desired=VY)
    mx = rot_z(v=VX, angle=PI)
    assert_array_almost_equal(actual=mx, desired=-VX)
    my = rot_z(v=VY, angle=PI)
    assert_array_almost_equal(actual=my, desired=-VY)

    rx = rot_z(v=VX, angle=PI / 2)
    assert_array_almost_equal(actual=rx, desired=VY)


def test_polarization_only():
    for phi in np.linspace(-PI, PI, 100):
        t = compile_homogeneous_transformation(
            azimuth_rad=0.0,
            zenith_rad=0.0,
            polarization_angle_rad=phi,
        )
        assert_no_translation(t=t)

        p = homtra.transform_orientation(t=t, d=V_PROPAGATION)
        assert_array_almost_equal(actual=p, desired=V_PROPAGATION)

        e = homtra.transform_orientation(t=t, d=V_E_FIELD)
        assert_almost_equal(actual=norm(e), desired=1)
        assert_array_almost_equal(
            actual=e, desired=rot_z(v=V_E_FIELD, angle=phi)
        )

        b = homtra.transform_orientation(t=t, d=V_B_FIELD)
        assert_almost_equal(actual=norm(b), desired=1)
        assert_array_almost_equal(
            actual=b, desired=rot_z(v=V_B_FIELD, angle=phi)
        )


def test_zenith_only():
    for zenith in np.linspace(0, PI / 2, 100):
        t = compile_homogeneous_transformation(
            azimuth_rad=0.0,
            zenith_rad=zenith,
            polarization_angle_rad=0.0,
        )
        assert_no_translation(t=t)

        p = homtra.transform_orientation(t=t, d=V_PROPAGATION)
        assert_array_almost_equal(
            actual=p, desired=rot_y(v=V_PROPAGATION, angle=zenith)
        )

        e = homtra.transform_orientation(t=t, d=V_E_FIELD)
        assert_array_almost_equal(
            actual=e, desired=rot_y(v=V_E_FIELD, angle=zenith)
        )

        assert_transform_orientation_has_no_effect(t=t, v=V_B_FIELD)


def test_azimuth_only():
    """
    This must not rotate anything
    """
    for azimuth in np.linspace(-PI, PI, 100):
        t = compile_homogeneous_transformation(
            azimuth_rad=azimuth,
            zenith_rad=0.0,
            polarization_angle_rad=0.0,
        )
        assert_no_translation(t=t)
        assert_transform_orientation_has_no_effect(t, VX)
        assert_transform_orientation_has_no_effect(t, VY)
        assert_transform_orientation_has_no_effect(t, VZ)


def test_azimuth_with_constant_zenith():
    ZENITH = PI / 4

    for azimuth in np.linspace(-PI, PI, 100):
        t = compile_homogeneous_transformation(
            azimuth_rad=azimuth,
            zenith_rad=ZENITH,
            polarization_angle_rad=0.0,
        )
        assert_no_translation(t=t)

        p = homtra.transform_orientation(t=t, d=V_PROPAGATION)
        assert_almost_equal(actual=p[2], desired=1 / np.sqrt(2))


def test_pointing_is_independent_of_polarization():
    NUM_POLA = 10
    for azimuth in np.linspace(-PI, PI, 10):
        for zenith in np.linspace(0, PI / 2, 10):
            poinings = []
            for pola in np.linspace(-PI, PI, NUM_POLA):
                t = compile_homogeneous_transformation(
                    azimuth_rad=azimuth,
                    zenith_rad=zenith,
                    polarization_angle_rad=pola,
                )
                assert_no_translation(t=t)

                p = homtra.transform_orientation(t=t, d=V_PROPAGATION)
                poinings.append(p)
            poinings = np.array(poinings)

            assert poinings.shape[0] == NUM_POLA
            assert poinings.shape[1] == 3
            assert_almost_equal(actual=np.std(poinings[:, 0]), desired=0.0)
            assert_almost_equal(actual=np.std(poinings[:, 1]), desired=0.0)
            assert_almost_equal(actual=np.std(poinings[:, 2]), desired=0.0)


def test_specific_case_1():
    zenith = PI / 4
    azimuth = PI

    t = compile_homogeneous_transformation(
        azimuth_rad=azimuth,
        zenith_rad=zenith,
        polarization_angle_rad=0.0,
    )
    assert_no_translation(t=t)

    p = homtra.transform_orientation(t=t, d=V_PROPAGATION)
    e = homtra.transform_orientation(t=t, d=V_E_FIELD)
    b = homtra.transform_orientation(t=t, d=V_B_FIELD)

    assert_array_almost_equal(actual=p, desired=[-ISQRT2, 0, ISQRT2])
    assert_array_almost_equal(actual=e, desired=[ISQRT2, 0, ISQRT2])
    assert_array_almost_equal(actual=b, desired=V_B_FIELD)


def test_specific_case_2():
    zenith = PI / 4
    azimuth = PI
    pola = PI / 2

    t = compile_homogeneous_transformation(
        azimuth_rad=azimuth,
        zenith_rad=zenith,
        polarization_angle_rad=pola,
    )
    assert_no_translation(t=t)

    p = homtra.transform_orientation(t=t, d=V_PROPAGATION)
    e = homtra.transform_orientation(t=t, d=V_E_FIELD)
    b = homtra.transform_orientation(t=t, d=V_B_FIELD)

    assert_array_almost_equal(actual=p, desired=[-ISQRT2, 0, ISQRT2])
    assert_array_almost_equal(actual=e, desired=[0, 1, 0])
    assert_array_almost_equal(actual=b, desired=[-ISQRT2, 0, -ISQRT2])
