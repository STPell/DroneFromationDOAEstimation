"""
    Provide support for delta-Angle thresholding. Produce some figures to explore the delta angle space.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   06/04/22
"""

import PointGenerator as pg
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np

# Import stuff for inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import cycler as cy

LINE_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
           'tab:olive', 'tab:cyan']

MARKER_FORMATS = ['o', 's', "*"]
format_cycler = cy.cycler(linestyle='-') * cy.cycler(marker=MARKER_FORMATS) * cy.cycler(color=LINE_COLOURS)


def set_up_latex():
    matplotlib.use("pgf")
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "pgf.preamble": [
            "\\usepackage{units}",  # load additional packages
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
            "\\usepackage{gensymb}",
        ]
    })


def est_angle_difference(measurement_separation, est_relative_bearing, target_dist):
    """
       Evaluate the expected difference in angle measurements to a target at range target_dist metres. Each measurement
       is taken measurement_separation metres apart and the target is at a relative bearing est_relative_bearing to the
       line between the two measurement points (measurement axis)
    """
    bearing = np.deg2rad(est_relative_bearing)
    x, y = target_dist * np.sin(bearing), target_dist * np.cos(bearing)
    a1 = pg.relative_angle_2d(-measurement_separation / 2, 0, x, y)
    a2 = pg.relative_angle_2d(measurement_separation / 2, 0, x, y)
    da = np.abs(a1 - a2)
    if da > 180:
        da = 360 - da
    return da


def generate_angle_difference_curve(d):
    MAX_R = int(1e4)
    R = list(range(100, MAX_R + 100, 100))

    delta_angles = []
    d1 = (-d / 2, 0)
    d2 = (d / 2, 0)
    for r in R:

        a1 = pg.relative_angle_2d(*d1, 0, r)
        a2 = pg.relative_angle_2d(*d2, 0, r)
        delta_angles.append(a1 - a2)

    print(f"ANGLE_DIFF_{d:.0f}M_DIST = {R}[::-1]")
    print(f"ANGLE_DIFF_{d:.0f}M_ANGLE_DIFF = {delta_angles}[::-1]")


def test_difference_in_angles_two_ensemble_multiple_distances():
    MAX_R = int(0.6e4)
    D = range(10, 41, 10)
    R = list(range(1, MAX_R + 100, 1))

    # set_up_latex()

    fig, ax = plt.subplots(constrained_layout=False)
    ax_ins = inset_axes(ax, width="50%", height="35%", loc="center right", borderpad=1.5)

    bearing = 0

    for d in D:
        delta_angles = []
        d1 = (-d / 2, 0)
        d2 = (d / 2, 0)
        for r in R:
            x, y = r * np.sin(bearing), r * np.cos(bearing)
            a1 = pg.relative_angle_2d(*d1, x, y)
            a2 = pg.relative_angle_2d(*d2, x, y)
            delta_angles.append(np.abs(a1 - a2))

        ax.plot(R, delta_angles, "-", label=f"$D=$ {d}\\,m")
        ax_ins.plot(R, delta_angles, "-", label=f"{d} m")

    ax.legend()
    ax.set_xlim([0, MAX_R])
    ax.set_ylim([0, 23])
    ax.set_xlabel("$R$ (m)")
    # ax.set_ylabel("Difference in angle (\N{DEGREE SIGN})")
    ax.set_ylabel("$\Delta{}A$ ($^\circ$)")
    # mark_inset(ax, ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")

    ax_ins.set_xlim([4800, 5800])
    ax_ins.set_ylim([0.0, 0.5])

    fig.savefig("difference_in_angle_per_distance.pgf")


def test_difference_in_angles_two_ensemble_multiple_angles():
    MAX_R = int(0.6e4)
    D = [40] #range(10, 41, 10)
    R = list(range(100, MAX_R + 100, 100))

    # set_up_latex()

    fig, ax = plt.subplots(constrained_layout=False)
    ax_ins = inset_axes(ax, width="50%", height="35%", loc="center right", borderpad=1.5)
    plt.rc('axes', prop_cycle=format_cycler)

    bearings = [0, 15, 30, 45, 60, 75, 90]

    for b in bearings:
        bearing = np.deg2rad(b)
        for d in D:
            delta_angles = []
            d1 = (-d / 2, 0)
            d2 = (d / 2, 0)
            for r in R:
                x, y = r * np.sin(bearing), r * np.cos(bearing)
                a1 = pg.relative_angle_2d(*d1, x, y)
                a2 = pg.relative_angle_2d(*d2, x, y)
                delta_angles.append(np.abs(a1 - a2))

            ax.plot(R, delta_angles, "-o", label=f"$D=${d}\\,m, bearing={b}\N{DEGREE SIGN}")
            ax_ins.plot(R, delta_angles, "-o", label=f"{d} m")

    ax.legend()
    ax.set_xlim([0, MAX_R])
    ax.set_ylim([0, 23])
    ax.set_xlabel("Distance to Target (m)")
    # ax.set_ylabel("Difference in angle (\N{DEGREE SIGN})")
    ax.set_ylabel("Difference in angle ($^\circ$)")
    # mark_inset(ax, ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")

    ax_ins.set_xlim([4800, 5800])
    ax_ins.set_ylim([0.0, 0.5])

    # fig.savefig("difference_in_angle_per_distance.pgf")


def test_difference_in_angles_two_ensemble_all_angles():
    D = 40
    R = [100, 500, 1000, 5000]
    # set_up_latex()

    fig, ax = plt.subplots(constrained_layout=False)
    bearings = np.arange(0, 360, 0.01)

    d1 = (-D / 2, 0)
    d2 = (D / 2, 0)

    for r in R:
        delta_angles = []
        for b in bearings:
            bearing = np.deg2rad(b)
            x, y = r * np.sin(bearing), r * np.cos(bearing)
            a1 = pg.relative_angle_2d(*d1, x, y)
            a2 = pg.relative_angle_2d(*d2, x, y)
            da = np.abs(a1 - a2)
            if da > 180:
                da = 360 - da
            delta_angles.append(da)

        ax.plot(bearings, delta_angles, "-", label=f"$R=$ {r}\\,m")

    plt.legend(loc="upper right")
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 23])
    ax.set_xlabel("Target Bearing ($^\circ$)")
    ax.set_ylabel("$\Delta{}A$ ($^\circ$)")
    fig.savefig("difference_in_angle_per_direction.pgf")


def plot_required_radii():
    MIN_SEP = [5, 10]
    N = list(range(3, 30))
    plt.figure()

    for min_sep in MIN_SEP:
        rs = []
        for n in N:
            theta = 2 * math.pi / n
            r = min_sep / (2 * np.sin(theta / 2))
            rs.append(r)
        plt.plot(N, rs, "-o", label=f"$d_{{safe}}=${min_sep} m")
    plt.xlabel("Number of Drones")
    plt.xlim([min(N), max(N)])
    plt.ylabel("Radius (m)")
    plt.title("Minimum formation radius to maintain safe separation ($d_{safe}$) between drones")
    plt.legend()


def test_difference_in_angles_circle_formation():
    R = list(range(4, 21, 2))
    NUM_DRONES = 5
    L = list(range(0, int(1e4), 10))

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for r in R:
        locs_x, locs_y, _ = pg.generate_points_uniform_on_circle_cart(NUM_DRONES, r)
        ensemble_est = []
        rmse = []
        for l in L:
            num, den = 0, 0
            v = []
            for i in range(NUM_DRONES):
                angle = np.deg2rad(pg.relative_angle_2d(locs_x[i], locs_y[i], 0, l))
                num += np.sin(angle)
                den += np.cos(angle)
                v.append(np.rad2deg(angle))
            ensemble_est.append(np.rad2deg(np.arctan2(num, den)))
            rmse.append(v)
        ax1.plot(L, ensemble_est, "-o", label=f"$r$={r}")
        rmse = np.array(rmse)
        ax2.scatter(np.repeat(L, NUM_DRONES), rmse.flatten(), label=f"$r$={r}")
    ax1.set_xlabel("Distance to Target (m)")
    ax1.set_ylabel("Error in ensemble estimate (\N{DEGREE SIGN})")
    fig1.legend()

    ax2.set_xlabel("Distance to Target (m)")
    ax2.set_ylabel("Error in individual estimates (\N{DEGREE SIGN})")
    ax2.set_xlim([0, 100])
    fig2.legend()


def main():
    # generate_angle_difference_curve(40)
    #
    # test_difference_in_angles_two_ensemble_multiple_distances()
    # test_difference_in_angles_two_ensemble_multiple_angles()
    # test_difference_in_angles_two_ensemble_all_angles()
    # plot_required_radii()
    # test_difference_in_angles_circle_formation()
    # plt.show()

    set_up_latex()
    test_difference_in_angles_two_ensemble_multiple_distances()
    test_difference_in_angles_two_ensemble_all_angles()


if __name__ == "__main__":
    main()
