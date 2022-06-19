"""
    A class to handle converting between a drones reference view and the global coordinate reference.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   14/03/22
"""


import numpy as np


class OrientableDrone:

    def __init__(self, x, y, z=0, orientation=0, ant=None):
        """Orientation is bearing relative to true north in degrees."""
        self.pos = (x, y, z)
        self.antenna = ant
        self.orientation = np.mod(orientation, 360)  # Orientation relative to true north in degrees

    def get_relative_bearing(self, true_bearing):
        """Return the bearing the the drone would measure from its own true north. Converts a bearing measured to the
           global bearing reference frame to the drone's reference frame."""
        return np.mod(true_bearing - self.orientation, 360)

    def get_true_bearing(self, relative_bearing):
        """Convers the relative bearing provided into the global bearing reference frame. Converts a bearing measured
           in this drone reference frame, to the global bearing reference frame."""
        return np.mod(relative_bearing + self.orientation, 360)

    def set_orientation(self, new_orientation):
        self.orientation = np.mod(new_orientation, 360)

    def get_orientation(self):
        return self.orientation

    def get_antenna(self):
        return self.antenna

    def get_3d_position(self):
        return self.pos

    def set_3d_position(self, x, y, z):
        self.pos = (x, y, z)

    def get_2d_position(self):
        return self.pos[0], self.pos[1]

    def set_2d_position(self, x, y):
        self.pos = (x, y, self.pos[2])

    def get_x_pos(self):
        return self.pos[0]

    def get_y_pos(self):
        return self.pos[1]

    def get_z_pos(self):
        return self.pos[2]

    def set_x_pos(self, x):
        self.pos= (x, self.pos[1], self.pos[2])

    def set_y_pos(self, y):
        self.pos = (self.pos[0], y, self.pos[2])

    def set_z_pos(self, z):
        self.pos = (self.pos[0], self.pos[1], z)


def test_orientable_drone():
    drone = OrientableDrone(90)
    print(drone.get_true_bearing(270))
    print(drone.get_relative_bearing(0))


if __name__ == "__main__":
    test_orientable_drone()
