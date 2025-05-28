import numpy as np
import spherical_coordinates


def screen_to_sky(x_m, focal_length_m):
    return -1.0 * np.arctan(x_m / focal_length_m)


def sky_to_screen(x_rad, focal_length_m):
    return -1.0 * np.tan(x_rad) * focal_length_m


def screen_area_to_sky_solid_angle(a_m2, focal_length_m):
    d_m = np.sqrt(a_m2)
    d_rad = screen_to_sky(d_m, focal_length_m)
    return d_rad**2


def screen_x_y_to_sky_az_zd(x_m, y_m, focal_length_m):
    cx = screen_to_sky(x_m, focal_length_m)
    cy = screen_to_sky(y_m, focal_length_m)
    return spherical_coordinates.cx_cy_to_az_zd(cx=cx, cy=cy)


def sky_az_zd_to_screen_x_y(azimuth_rad, zenith_rad, focal_length_m):
    cx, cy = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
    )
    x = sky_to_screen(cx, focal_length_m)
    y = sky_to_screen(cy, focal_length_m)
    return x, y
