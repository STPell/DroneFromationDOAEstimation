"""
    Allow for easy simulation of drone swarms by allowing swarms to be manipulated at the formation level.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   14/03/22

    Changelog:
        06/04/22 Add a way to retrieve previous bearing estimates
        06/04/22 Allow the formation to take the incremented orientation scheme
        26/04/22 Allow distance based SNR
"""

import numpy as np
import PointGenerator as pg
import OrientableDrone as od
import sectoredAntenna as sa
import doa_est_tsls as tsls
import SignalGenerator as sg
import samUtilities as su
import enum

import matplotlib.pyplot as plt


@enum.unique
class DroneSwarmOrientations(enum.Enum):
    RANDOM = 1
    UNIFORM = 2
    INCREMENTED = 3
    PARTITION = 4


class DroneSwarm(object):
    tx_x = 0
    tx_y = 0

    formation_centre_x = 0
    formation_centre_y = 0

    def __init__(self):
        self.drones = []
        self.estimated_drone_bearings = []  # Keep a history of the estimates of the position. The first element is the
                                            # most recent bearing estimate
        self.last_estimated_direction = None

    def set_tx_loc(self, x, y):
        self.tx_x, self.tx_y = x, y

    def get_tx_loc(self):
        return self.tx_x, self.tx_y

    def get_formation_centre(self):
        return self.formation_centre_x, self.formation_centre_y

    def _set_formation_centre(self, x, y):
        self.formation_centre_x, self.formation_centre_y = x, y

    def add_drone(self, drone):
        self.drones.append(drone)

    def rotate_drones(self, theta):
        """Rotate the drone orientations indivdually, rather than pivoting the entire formation."""
        for drone in self.drones:
            drone.set_orientation(drone.get_orientation() + theta)

    def estimate_tx_bearing(self, snr=None):
        est_bearings = []
        for drone in self.drones:
            # TODO: Check if we need to mess with the orientation of the drones due to formation rotation
            true_bearing = pg.relative_bearing_2d(drone.get_x_pos(), drone.get_y_pos(), self.tx_x, self.tx_y)
            if snr is not None:
                sector_rx_powers, noise_var = sg.generate_rx_powers_fixed(snr, drone.get_x_pos(),
                                                                          drone.get_y_pos(), drone.get_antenna(),
                                                                          self.tx_x, self.tx_y,
                                                                          drone.get_relative_bearing(true_bearing))
            else:
                sector_rx_powers, noise_var = sg.generate_rx_powers(drone.get_x_pos(), drone.get_y_pos(),
                                                                    drone.get_antenna(), self.tx_x, self.tx_y,
                                                                    drone.get_relative_bearing(true_bearing))
            est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                     L=3)
            est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

        self.estimated_drone_bearings.insert(0, est_bearings)
        self.last_estimated_direction = self._combine_estimate_bearings(est_bearings)
        return self.last_estimated_direction

    def _combine_estimate_bearings(self, estimates):
        numerator, denominator = 0, 0
        for est in estimates:
            numerator += np.sin(np.deg2rad(est))
            denominator += np.cos(np.deg2rad(est))
        return np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)

    def move_swarm(self, dir, dist):
        """Move the swarm dist meters on a bearing of dir degrees"""
        dir = np.deg2rad(dir)
        dx = dist * np.sin(dir)
        dy = dist * np.cos(dir)
        for drone in self.drones:
            drone.set_x_pos(drone.get_x_pos() + dx)
            drone.set_y_pos(drone.get_y_pos() + dy)

        old_centre = self.get_formation_centre()
        self._set_formation_centre(old_centre[0] + dx, old_centre[1] + dy)

    def rotate_swarm(self, theta):
        """Rotate the swarm theta degrees"""
        theta = np.deg2rad(theta)
        centre_x, centre_y = self.get_formation_centre()
        for drone in self.drones:
            x, y = drone.get_2d_position()
            x = x - centre_x
            y = y - centre_y
            x_new = (x * np.cos(theta)) - (y * np.sin(theta))
            y_new = (x * np.sin(theta)) + (y * np.cos(theta))
            drone.set_2d_position(x_new + centre_x, y_new + centre_y)

    def set_swarm_position(self, new_swarm_x, new_swarm_y):
        """Change the swarms centre position and move all drones to this new position"""
        form_centre = self.get_formation_centre()
        for drone in self.drones:
            x, y = drone.get_2d_position()
            x = x - form_centre[0] + new_swarm_x
            y = y - form_centre[1] + new_swarm_y
            drone.set_2d_position(x, y)

        self._set_formation_centre(new_swarm_x, new_swarm_y)

    def get_bearing_last_estimates(self):
        """Return the most recent individual bearing estimates for the swarm"""
        return self.estimated_drone_bearings[0]

    def use_incremented_drone_orientation(self):
        num_drones = len(self.drones)
        ant_width = self.drones[0].get_antenna().beam_width(0)
        for i in range(num_drones):
            self.drones[i].set_orientation(i * ant_width / num_drones)


class CircularDroneFormation(DroneSwarm):

    def __init__(self, x, y, r, num_drones, num_sectors, a_s):
        DroneSwarm.__init__(self)
        self.formation_radius = r
        self._set_formation_centre(x, y)
        drone_offset_x, drone_offset_y, _ = pg.generate_points_uniform_on_circle_cart(num_drones, r)
        for i in range(num_drones):
            ant = sa.IdealSectoredAntenna(num_sectors, a_s)
            drone = od.OrientableDrone(x + drone_offset_x[i], y + drone_offset_y[i], ant=ant)
            self.add_drone(drone)

    def plot_swarm_position(self, fig, colour='tab:blue', arrow_len=2, label=None):
        """Plot the swarm and its pointing direction onto a figure"""
        locs_x = []
        locs_y = []
        for drone in self.drones:
            x, y = drone.get_x_pos()[0], drone.get_y_pos()[0]
            drone_or = np.deg2rad(drone.get_orientation())
            fig.gca().arrow(x, y, arrow_len * np.sin(drone_or), arrow_len * np.cos(drone_or), color=colour, width=0.2,
                            fill=False)
            locs_x.append(x)
            locs_y.append(y)

        fig.gca().scatter(locs_x, locs_y, color=colour, label=label)
        c = plt.Circle(self.get_formation_centre(), self.formation_radius, color=colour,
                       fill=False, linestyle="--")
        fig.gca().add_artist(c)

    def plot_swarm_est_directions(self, fig, colour='tab:blue', arrow_len=2):
        latest_bearing_est = self.get_bearing_last_estimates()
        for i in range(len(self.drones)):
            drone = self.drones[i]
            x, y = drone.get_x_pos()[0], drone.get_y_pos()[0]
            point_dir = np.deg2rad(latest_bearing_est[i])
            fig.gca().arrow(x, y, arrow_len * np.sin(point_dir), arrow_len * np.cos(point_dir), color=colour, width=0.2,
                            fill=False, linestyle="--")

    def plot_swarm_overall_direction(self, fig, colour="tab:blue", arrow_len=2):
        point_dir = np.deg2rad(self.last_estimated_direction)
        fig.gca().arrow(*self.get_formation_centre(), arrow_len * np.sin(point_dir), arrow_len * np.cos(point_dir),
                        color=colour, width=0.2, fill=False)


class LinearDroneFormation(DroneSwarm):

    def __init__(self, x, y, d, init_angle, num_drones, num_sectors, a_s):
        DroneSwarm.__init__(self)
        self.drone_spacing = d
        self._set_formation_centre(x, y)
        self.line_angle = init_angle
        init_angle = np.deg2rad(init_angle)
        for i in range(num_drones):
            j = i - (num_drones / 2) + 0.5
            loc_x = x + ((j * d) * np.sin(init_angle))
            loc_y = x + ((j * d) * np.cos(init_angle))
            ant = sa.IdealSectoredAntenna(num_sectors, a_s)
            drone = od.OrientableDrone(loc_x, loc_y, ant=ant)
            self.add_drone(drone)

    def plot_swarm_position(self, fig, colour='tab:blue', arrow_len=2, label=None):
        """Plot the swarm and its pointing direction onto a figure"""
        locs_x = []
        locs_y = []
        for drone in self.drones:
            x, y = drone.get_x_pos(), drone.get_y_pos()
            drone_or = np.deg2rad(drone.get_orientation())
            fig.gca().arrow(x, y, arrow_len * np.sin(drone_or), arrow_len * np.cos(drone_or), color=colour, width=0.2,
                            fill=False)
            locs_x.append(x)
            locs_y.append(y)
        fig.gca().scatter(locs_x, locs_y, color=colour, label=label)


def test_circular_swarm():
    fig = plt.figure()
    s = CircularDroneFormation(10, 20, 10, 12, 6, 0.4)
    s.plot_swarm_position(fig, arrow_len=2)
    s.move_swarm(45, 20)
    s.rotate_drones(45)
    s.plot_swarm_position(fig, colour='tab:orange', arrow_len=2)
    s.rotate_swarm(15)
    s.rotate_drones(-90)
    s.plot_swarm_position(fig, colour='tab:green', arrow_len=2)
    s.set_swarm_position(-50, -50)
    s.plot_swarm_position(fig, colour='tab:purple', arrow_len=2)
    s.move_swarm(45, np.sqrt(2 * (50 ** 2)))
    s.plot_swarm_position(fig, colour='tab:brown', arrow_len=2)


def test_linear_swarm():
    fig = plt.figure()
    s = LinearDroneFormation(0, 0, 10, 90, 12, 6, 0.4)
    s.plot_swarm_position(fig, label="Even")
    s = LinearDroneFormation(0, 0, 10, 90, 11, 6, 0.4)
    s.plot_swarm_position(fig, colour="tab:orange", label="Odd")

    s = LinearDroneFormation(0, 0, 10, 90, 11, 6, 0.4)
    s.rotate_swarm(-45)
    s.plot_swarm_position(fig, colour="tab:green", label="Odd, rotated -90")

    for i in range(4):
        s.rotate_drones(45)
        s.move_swarm(45, 10)
        s.plot_swarm_position(fig, colour="tab:purple", label=f"Odd, rotated -90, advance on 45 deg step:{i}")
    plt.legend()

    s.move_swarm(45, -15)
    s.plot_swarm_position(fig, colour="tab:brown")

    s.set_swarm_position(-10, -10)
    s.rotate_swarm(45)
    s.plot_swarm_position(fig, colour="tab:red")


if __name__ == "__main__":
    test_circular_swarm()
    test_linear_swarm()
    plt.show()