"""
    A 'library' to generate points in space around the origin

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   09/08/21
"""

import numpy as np
import math


def convert_cart_to_sph(x, y, z):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    phi = np.arctan2(y, x)
    theta = np.arccos(np.divide(z, r))
    return r, phi, theta


def convert_sph_to_cart(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def generate_points_in_ball(n: int, r: float):
    """Generate n random points in spherical coordinates from the ball of radius r centred at the origin"""
    phi_ = np.random.uniform(0, 1, (n, 1)) * 2 * math.pi
    theta_ = np.arccos(2 * np.random.uniform(0, 1, (n, 1)) - 1)
    dists_ = np.cbrt(np.random.uniform(0, 1, (n, 1))) * r
    return dists_, phi_, theta_


def generate_points_in_ball_cart(n: int, r: float):
    """Generate n random points in Cartesian coordinates from the ball of radius r centred at the origin"""
    r, phi, theta = generate_points_in_ball(n, r)
    return convert_sph_to_cart(r, phi, theta)


def generate_points_on_sphere_cart(n: int, r: float):
    """Generate n random points in cartesian coordinates from the surface of a sphere with radius r centred at the
       origin."""
    u = np.random.uniform(-1, 1, (n, 1))
    theta = np.random.uniform(0, 2 * math.pi, (n, 1))
    u_squared = np.square(u)
    x = r * np.sqrt(1 - u_squared) * np.cos(theta)
    y = r * np.sqrt(1 - u_squared) * np.sin(theta)
    z = r * u
    return x, y, z


def generate_points_on_sphere(n: int, r: float):
    """Generate n random points in spherical coordinates from the surface of a sphere with radius r centred at the
       origin."""
    return convert_cart_to_sph(*generate_points_on_sphere_cart(n, r))


def generate_points_uniform_on_circle(n: int, r: float):
    """Generate n points in spherical coordinates uniformly distributed around the x-y circle with radius r, centred at
       the origin."""
    theta = math.pi * np.ones((n, 1)) / 2  # Theta is 90deg for all values
    phi = np.arange(n).reshape((n, 1)) * 2 * math.pi / n
    r = r * np.ones((n, 1))
    return r, phi, theta


def generate_points_uniform_on_circle_cart(n: int, r: float):
    """Generate n points in cartesian coordinates uniformly distributed around the x-y circle with radius r, centred at
       the origin."""
    r, phi, theta = generate_points_uniform_on_circle(n, r)
    return convert_sph_to_cart(r, phi, theta)


def relative_bearing_2d(x1, y1, x2, y2):
    """
       Calculate the bearing in degrees between two objects in the global coordinate system (ignoring the orientation of
       the either object at the coordinates.
    """
    bearing = np.arctan2(x2 - x1, y2 - y1)
    if bearing < 0:
        bearing += 2 * math.pi
    return np.rad2deg(bearing)


def relative_angle_2d(x1, y1, x2, y2):
    """
       Calculate the bearing in degrees between two objects in the global coordinate system (ignoring the orientation of
       the either object at the coordinates.
    """
    bearing = np.arctan2(x2 - x1, y2 - y1)
    return np.rad2deg(bearing)


def generate_point_uniform_in_disc_cart(n, r_1, r_0=0):
    x, y = [], []
    while len(x) < n:
        x_0, y_0 = np.random.uniform(-1.5 * r_1 , 1.5 * r_1, 2)
        if r_0 <= np.sqrt((x_0 ** 2.0) + (y_0 ** 2.0)) <= r_1:
            x.append(x_0)
            y.append(y_0)
    return x, y
