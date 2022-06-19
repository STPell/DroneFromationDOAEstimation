"""
    A library implementing an approximation of a sectored antenna based on the model in J. Werner et al., "Sectorized
    Antenna-based DoA Estimation and Localization: Advanced Algorithms and Measurements," in IEEE Journal on Selected
    Areas in Communications, vol. 33, no. 11, pp. 2272-2286, Nov. 2015, doi: 10.1109/JSAC.2015.2430292.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   15/12/21
"""

import numpy as np
import math


class SectoredAntenna(object):

    def __init__(self, num_sectors, max_gains, beam_widths, beam_directions, equal_beamwidths=False):
        """Create a sectored antenna model object. This model is based on the model documented in [1]. This model has a
           few limitations, the main being that it assumes that the gain of each sector is only dependent on the azimuth
           of the sector and does not take into account any directionality of the antenna in the elevation dimension.

           Inputs:
                - num_sectors:     the number of sectors of the model antenna
                - max_gains:       the maximum gain of each sector (dB)
                - beam_widths:     the width of the main lobe of each sector (degrees)
                - beam_directions: the direction in which each sector has its maximum gain ([0-360) degrees)

            -----------------------
            References:
                [1] J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
                    Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286,
                    Nov. 2015, doi: 10.1109/JSAC.2015.2430292.
        """
        if len(max_gains) != num_sectors:
            raise ValueError("To few gains have been specified based on the number of sectors selected")

        if len(beam_widths) != num_sectors:
            raise ValueError("To few beam widths have been specified based on the number of sectors selected")

        if len(beam_directions) != num_sectors:
            raise ValueError("To few beam directions have been specified based on the number of sectors selected")

        self.M = num_sectors
        self.alpha = np.power(10, np.divide(max_gains, 10))
        self.beta = np.deg2rad(beam_widths)
        self.theta = np.deg2rad(beam_directions)
        self.equal_beamwidths = equal_beamwidths

    @staticmethod
    def _m_func(phi):
        return np.mod(phi + math.pi, 2 * math.pi) - math.pi

    def sector_gain(self, sector, dir, dB=False):
        """
           Return the gain of the sector antenna in the provided direction in decibels or natural value. Sectors are
           indexed from 0. Direction is in degrees.
        """
        if sector > self.M or sector < 0:
            raise IndexError("Desired sector out of index")

        dir = np.mod(dir, 360)
        x = np.power(SectoredAntenna._m_func(np.deg2rad(dir) - self.theta[sector]), 2) / np.power(self.beta[sector], 2)
        gain = self.alpha[sector] * np.exp(-x)
        if dB:
            return 10 * np.log10(gain)
        else:
            return gain

    def num_sectors(self):
        return self.M

    def sector_direction(self, sector, rad=False):
        """Returns the direction at which the sector has its maximum gain value in degrees"""
        if sector > self.M or sector < 0:
            raise IndexError("Desired sector out of index")
        else:
            if rad:
                return self.theta[sector]
            else:
                return np.rad2deg(self.theta[sector])

    def has_equal_beamwidths(self):
        return self.equal_beamwidths

    def beam_width(self, sector, rad=False):
        """Return the beam width of the desired sector in radians or degrees"""
        if rad:
            return self.beta[sector]
        else:
            return np.rad2deg(self.beta[sector])


class IdealSectoredAntenna(SectoredAntenna):

    def __init__(self, num_sectors=6, a_s=None):
        """Create an ideal num_sector-sectored antenna. If a_s is not specified, the 3dB beam width is set to delta_v"""
        delta_v = 360 / num_sectors

        if a_s is None:
            # Calculate the beam-width such that each sectors 3dB bandwidths do not overlap.
            beam_width = np.sqrt(-np.power(IdealSectoredAntenna._m_func(np.deg2rad(delta_v / 2)), 2) / np.log(0.5))
            print(beam_width)
        else:
            beam_width = 2 * np.pi / (num_sectors * np.sqrt(np.log(1 / a_s)))

        beam_dirs = np.arange(0, 360, delta_v)
        beam_widths = [np.rad2deg(beam_width) for i in range(num_sectors)]
        max_gains = [0 for i in range(num_sectors)]
        super().__init__(num_sectors, max_gains, beam_widths, beam_dirs, equal_beamwidths=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.arange(0, 360, 0.01)
    for M in [3, 4, 6, 8, 20]:
        fig = plt.figure()
        plt.suptitle(f"Approximation of the beam-pattern of an ideal {M}-sectored antenna")
        ax = fig.add_subplot(1, 1, 1, projection='polar')

        ant = IdealSectoredAntenna(M, a_s=0.4)
        for sector in range(M):
            y = []
            for i in range(len(x)):
                y.append(ant.sector_gain(sector, x[i], dB=True))
            ax.plot(np.deg2rad(x), y, label=f"Sector {sector}")

        plt.legend()
        plt.ylim([-10, 0])

    # Test non-ideal antenna
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ant = SectoredAntenna(2,                                       # Number of beams
                          10 * np.log10([1, 0.9]),                 # Attenuation
                          [np.rad2deg(1 / 2), np.rad2deg(1 / 3)],  # Beam Width
                          [0, 340])                                # Beam Direction
    for sector in range(ant.num_sectors()):
        y = []
        for i in range(len(x)):
            y.append(ant.sector_gain(sector, x[i], dB=True))
        ax.plot(np.deg2rad(x), y, label=f"Sector {sector}")

    plt.legend()
    plt.ylim([-10, 0])

    # Test non-ideal antenna
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ant = SectoredAntenna(12,
                          10 * np.log10([0.77, 0.90, 1.00, 0.98, 0.93, 0.92, 0.92, 0.93, 0.98, 1.00, 0.90, 0.77]),
                          np.rad2deg([0.64, 0.61, 0.62, 0.58, 0.52, 0.53, 0.53, 0.52, 0.58, 0.62, 0.61, 0.64]),
                          np.mod([-47.9, -40.6, -29.0, -17.6, -9.5, -1.5, 1.5, 9.5, 17.6, 29, 40.6, 47.9], 360))
    for sector in range(ant.num_sectors()):
        y = []
        for i in range(len(x)):
            y.append(ant.sector_gain(sector, x[i], dB=True))
        ax.plot(np.deg2rad(x), y, label=f"Sector {sector}")

    plt.legend()
    plt.ylim([-10, 0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    a_s_values = [0.1, 0.2, 0.4, 0.6289062225842643]
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
               'tab:olive', 'tab:cyan']
    for i in range(len(a_s_values)):
        a_s = a_s_values[i]

        ant = IdealSectoredAntenna(4, a_s=a_s)
        for sector in range(ant.num_sectors()):
        # sector = 0
            y = []
            for j in range(len(x)):
                y.append(ant.sector_gain(sector, x[j], dB=True))
            ax.plot(np.deg2rad(x), y, label=f"$a_s$ = {a_s:.2f}, Sector {sector}", color=colours[i])

    plt.ylim([-10, 0])

    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    M = 4
    for a_s in [0.003151142260559449]:
        ant = IdealSectoredAntenna(M, a_s=a_s)
        for sector in range(M):
            y = []
            for i in range(len(x)):
                y.append(ant.sector_gain(sector, x[i], dB=True))
            ax.plot(np.deg2rad(x), y, label=f"Sector {sector}")

        plt.legend()
        plt.ylim([-10, 0])

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    M = 4
    for a_s in [0.03916]:
        ant = IdealSectoredAntenna(M, a_s=a_s)
        for sector in range(M):
            y = []
            for i in range(len(x)):
                y.append(ant.sector_gain(sector, x[i], dB=True))
            ax.plot(np.deg2rad(x), y, label=f"Sector {sector}")

        plt.legend()
        plt.ylim([-10, 0])

    plt.show()

