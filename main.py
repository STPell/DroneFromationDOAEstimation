"""
   A playground to simulate an RSSI-based Direction of Arrival (DoA) system.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   04/05/2021
"""

import samUtilities as su
import PointGenerator as pg
import numpy as np
import matplotlib.pyplot as plt
import sectoredAntenna as sa
import os
import doa_est_tsls as tsls
import statistics as stats
import cycler as cy
import rich.progress as rp
import OrientableDrone as od
from datetime import datetime
import multiprocessing as multi
from SignalGenerator import *

NUM_PROC = 8                # Number of processors available for parallelism
AREA_BOUND = 2000           # m, Length of a single side of the simulated area
TX_AREA_BOUND = 2000        # m, Length of a single side of the sub-area the transmitter can be located in
TX_POWER_DB = 10            # dBm, Transmit power of the transmitter we are trying to locate
FREQUENCY = 2.45e9          # Hz, Transmit frequency
SNR = 5                     # dB, the signal to noise ratio
SAMPLES_PER_SECTOR = 100    # Number of samples per sector of the system
NUM_REPETITIONS = int(1e5)  # The number of repetitions to complete
SAMPLING_BANDWIDTH = 2.5e6  # Hz, Sampling bandwidth
RX_NOISE_FIGURE = 8         # dB, the noise figure of the receiver

NUM_SECTOR_VALUES = [6, 8, 10, 12]
SNR_VALUES = list(range(0, 15, 2))
NUM_DRONE_VALUES = [3, 5, 7, 9, 11, 13, 15]
NOISE_VAR_ERRS = [0.1, 0.5, 1.0, 1.5, 2, 2.5, 3]

FIGURES = {"abs_err": 1, "ecdf": 2, "ecdf_zoomed": 3}

PLOT_LINE_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:gray', 'tab:olive', 'tab:cyan']
PLOT_LINE_FORMATS = ['.', 'x', '+']
PLOT_FORMAT_CYCLER = cy.cycler(marker=PLOT_LINE_FORMATS) * cy.cycler(color=PLOT_LINE_COLOURS)


def maxE_bearing_est(antenna, sector_powers, N=2):
    """
       Use the a variation on the Maximum Energy (maxE) algorithm to estimate the direction of the transmitter. The maxE
       algorithm estimates the direction by taking the weighted average of the N strongest sectors.

       TODO: Find which paper I took this from
    """
    largest_N = np.argpartition(sector_powers, -N)[-N:]

    weighted_sum = 0
    total_weights = 0
    for sector in largest_N:
        sector_weight = su.db_to_nat(sector_powers[sector])
        weighted_sum += sector_weight * antenna.sector_direction(sector)
        total_weights += sector_weight

    est_direction = weighted_sum / total_weights
    return est_direction


def simulate_rmse_based_on_snr_and_num_sectors():
    res_file_name = datetime.now().strftime("results/effect_of_sectors/signal_strength_res_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,snr,num_repetitions,a_s,rmse,std_dev\n")

    a_s = 0.4
    snr_values = list(np.arange(-1, 20, 1))
    rmse_values = []
    for num_sectors in NUM_SECTOR_VALUES:
        for snr in snr_values:
            errs = []

            for repetition in range(NUM_REPETITIONS):
                # Generate random location of transmitter
                tx_loc_x = np.random.uniform(-TX_AREA_BOUND / 2, TX_AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-TX_AREA_BOUND / 2, TX_AREA_BOUND / 2)

                # Generate random location of receiver
                rx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                rx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                # rx_loc_y = np.random.uniform(0, -AREA_BOUND / 2)

                # Calculate the relative bearing of the transmitter as viewed from the receiver
                rel_bearing = pg.relative_bearing_2d(rx_loc_x, rx_loc_y, tx_loc_x, tx_loc_y)
                # print(f"Transmitter at {rel_bearing:.2f}\N{DEGREE SIGN} relative to the receiver")

                ant = sa.IdealSectoredAntenna(num_sectors=num_sectors, a_s=a_s)
                sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, rx_loc_x, rx_loc_y, ant,
                                                                       tx_loc_x, tx_loc_y, rel_bearing)

                # Try and estimate the bearing using the RSSI information
                # est_direction = maxE_bearing_est(ant, sector_rx_powers, N=2)
                est_direction, _ = tsls.tsls_bearing_est(ant, sector_rx_powers, su.nat_to_db(2 * noise_var), L=3)
                est_direction = np.mod(est_direction, 360)

                errs.append(min(360 - np.abs(est_direction - rel_bearing), np.abs(est_direction - rel_bearing)))

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_values.append((num_sectors, snr, rmse))
            res_file.write(f"{num_sectors},{snr},{NUM_REPETITIONS},{a_s},{rmse},{np.std(errs)}\n")
            res_file.flush()
    res_file.close()


def simulate_rmse_based_on_number_of_drones_fixed_snr():
    rmse_values = []
    num_sectors = 6
    for snr in SNR_VALUES:
        for num_drones in NUM_DRONE_VALUES:
            errs = []
            for repetition in range(NUM_REPETITIONS):
                # Generate random location of transmitter
                tx_loc_x = np.random.uniform(-TX_AREA_BOUND / 2, TX_AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-TX_AREA_BOUND / 2, TX_AREA_BOUND / 2)

                # Generate random location of rx centre
                rx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                rx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                shift_x, shift_y, shift_z = pg.generate_points_uniform_on_circle_cart(num_drones, 5)
                rx_locs_x = rx_loc_x + shift_x
                rx_locs_y = rx_loc_y + shift_y

                # Calculate the estimate bearing for each drone before combining them
                est_dirs = []
                for drone in range(num_drones):
                    # Calculate the relative bearing of the transmitter as viewed from the receiver
                    rel_bearing = pg.relative_bearing_2d(rx_locs_x[drone], rx_locs_y[drone], tx_loc_x, tx_loc_y)

                    ant = sa.IdealSectoredAntenna(num_sectors=num_sectors, a_s=0.4)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, rx_locs_x[drone], rx_locs_y[drone], ant,
                                                                           tx_loc_x, tx_loc_y, rel_bearing)

                    # Try and estimate the bearing using the RSSI information
                    est_dir, _ = tsls.tsls_bearing_est(ant, sector_rx_powers, su.nat_to_db(2 * noise_var), L=3)
                    est_dirs.append(np.mod(est_dir, 360))

                # Combine the estimates to create a better estimate.
                if len(est_dirs) > 1:
                    numerator, denominator = 0, 0
                    for est in est_dirs:
                        numerator += np.sin(np.deg2rad(est))
                        denominator += np.cos(np.deg2rad(est))
                    est_direction = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                else:
                    est_direction = est_dirs[0]

                rel_bearing = pg.relative_bearing_2d(rx_loc_x, rx_loc_y, tx_loc_x, tx_loc_y)
                errs.append(min(360 - np.abs(est_direction - rel_bearing), np.abs(est_direction - rel_bearing)))

            fig = plt.figure(FIGURES["abs_err"])
            plt.plot(errs, ".", label=f"{num_drones} sectors {snr}")

            fig = plt.figure(FIGURES["ecdf"])
            su.plot_ecdf(fig, errs, label=f"{num_drones} sectors {snr}")

            fig = plt.figure(FIGURES["ecdf_zoomed"])
            su.plot_ecdf(fig, errs, label=f"{num_drones} sectors {snr}")
            plt.xlim([0, 25])

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            print(f"{num_drones} @ {snr}: RMSE={rmse:.2f}")
            rmse_values.append((snr, num_drones, rmse))

    # Enable the legend on all plots
    for fig_number in FIGURES.values():
        fig = plt.figure(fig_number)
        plt.legend()

    plt.figure()
    for d in NUM_DRONE_VALUES:
        plt.plot([snr for snr, drones, rmse in rmse_values if drones == d],
                 [rmse for snr, drones, rmse in rmse_values if drones == d], ".",
                 label=f"{d} drones in swarm")
    # plt.plot([0, 5, 10, 15, 20], [8.5, 3.8, 2.1, 2.0, 1.9], "-o")

    plt.xlabel('SNR (dB)')
    plt.ylabel("Bearing Estimate Error (\N{DEGREE SIGN})")
    plt.legend()


def simulate_performance_of_combining_estimates():
    rmse_values = []
    num_sectors = 6
    for num_drones in NUM_DRONE_VALUES:
        errs = []
        for repetition in range(NUM_REPETITIONS):
            # Generate random location of transmitter
            tx_loc_x = np.random.uniform(-TX_AREA_BOUND / 2, TX_AREA_BOUND / 2)
            tx_loc_y = np.random.uniform(-TX_AREA_BOUND / 2, TX_AREA_BOUND / 2)

            # Generate random location of rx centre
            rx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
            rx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

            shift_x, shift_y, shift_z = pg.generate_points_uniform_on_circle_cart(num_drones, 5)
            rx_locs_x = rx_loc_x + shift_x
            rx_locs_y = rx_loc_y + shift_y

            # Calculate the estimate bearing for each drone before combining them
            est_dirs = []
            for drone in range(num_drones):
                # Calculate the relative bearing of the transmitter as viewed from the receiver
                rel_bearing = pg.relative_bearing_2d(rx_locs_x[drone], rx_locs_y[drone], tx_loc_x, tx_loc_y)

                ant = sa.IdealSectoredAntenna(num_sectors=num_sectors, a_s=0.4)
                sector_rx_powers, noise_var = generate_rx_powers(rx_locs_x[drone], rx_locs_y[drone], ant,
                                                                 tx_loc_x, tx_loc_y, rel_bearing)

                # Try and estimate the bearing using the RSSI information
                est_dir, _ = tsls.tsls_bearing_est(ant, sector_rx_powers, su.nat_to_db(2 * noise_var), L=3)
                est_dirs.append(np.mod(est_dir, 360))

            rel_bearing = pg.relative_bearing_2d(rx_loc_x, rx_loc_y, tx_loc_x, tx_loc_y)

            # Combine the estimates to create a better estimate.
            if len(est_dirs) > 1:
                numerator, denominator = 0, 0
                for est in est_dirs:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                dir_fisher_and_lewis = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(dir_fisher_and_lewis - rel_bearing),
                                           np.abs(dir_fisher_and_lewis - rel_bearing))

                dir_direction_median = stats.median(est_dirs)
                err_direction_median = min(360 - np.abs(dir_direction_median - rel_bearing),
                                           np.abs(dir_direction_median - rel_bearing))

                dir_direction_ave = np.average(est_dirs)
                err_direction_ave = min(360 - np.abs(dir_direction_ave - rel_bearing),
                                        np.abs(dir_direction_ave - rel_bearing))

                errs.append((err_fisher_and_lewis, err_direction_median, err_direction_ave))
            else:
                est_direction = est_dirs[0]
                errs.append(min(360 - np.abs(est_direction - rel_bearing), np.abs(est_direction - rel_bearing)))

        rmse = np.sqrt(np.mean(np.power(errs, 2), axis=0))
        std = np.std(errs, axis=0)
        print(f"{num_drones}: RMSE={rmse}, std={std}")
        rmse_values.append((num_drones, rmse, std))

    plt.figure()
    plt.plot([drones for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
             [rmse_1 for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values], ".", label="Fisher and Lewis")
    plt.plot([drones for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
             [rmse_2 for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values], ".", label="Median")
    plt.plot([drones for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
             [rmse_3 for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values], ".", label="Average")
    plt.title("Effect of combining method on resulting bearing")
    plt.xlabel('# Drones in Swarm')
    plt.ylabel("RMS Bearing Estimate Error (\N{DEGREE SIGN})")
    plt.legend()

    plt.figure()
    plt.errorbar([drones for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
                 [rmse_1 for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
                 yerr=[1.96 * std_1 / np.sqrt(NUM_REPETITIONS) for drones, rmse, (std_1, std_2, std_3) in rmse_values],
                 fmt=".", label="Fisher and Lewis", capsize=3)
    plt.errorbar([drones for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
                 [rmse_2 for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
                 yerr=[1.96 * std_2 / np.sqrt(NUM_REPETITIONS) for drones, rmse, (std_1, std_2, std_3) in rmse_values],
                 fmt=".", label="Median", capsize=3)
    plt.errorbar([drones for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
                 [rmse_3 for drones, (rmse_1, rmse_2, rmse_3), std in rmse_values],
                 yerr=[1.96 * std_3 / np.sqrt(NUM_REPETITIONS) for drones, rmse, (std_1, std_2, std_3) in rmse_values],
                 fmt=".", label="Average", capsize=3)
    plt.title("Effect of combining method on resulting bearing")
    plt.xlabel('# Drones in Swarm')
    plt.ylabel("RMS Bearing Estimate Error (\N{DEGREE SIGN})")
    plt.legend()



def investigate_effect_of_bearing():
    """See what parameters have an effect on RMSE as a function of bearing."""
    incoming_angle_values = list(range(0, 360, 1))
    NUM_SECTORS = 4
    snr = 5

    res_file_name = datetime.now().strftime("results/drone_orientation/res_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,snr,num_repetitions,a_s,doa,rmse,std_dev\n")

    for a_s in [0.03916]: #, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6289062225842643]:
        ant = sa.ant = sa.IdealSectoredAntenna(num_sectors=NUM_SECTORS, a_s=a_s)
        rmse_values = []
        for incoming_angle in incoming_angle_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS),
                                      description=f"a_s = {a_s:.2f}, Tx Bearing = {incoming_angle:.1f}\N{DEGREE SIGN}"):
                sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, 0, 0, ant, 30, 30, incoming_angle)
                est_direction, _ = tsls.tsls_bearing_est(ant, sector_rx_powers, su.nat_to_db(2 * noise_var), L=4)
                est_direction = np.mod(est_direction, 360)
                errs.append(min(360 - np.abs(est_direction - incoming_angle), np.abs(est_direction - incoming_angle)))

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            std = np.std(errs)
            rmse_values.append((rmse, std, len(errs)))
            res_file.write(f"{NUM_SECTORS},{snr},{len(errs)},{a_s},{incoming_angle},{rmse},{std}\n")

        ci = [1.960 * std / np.sqrt(n) for _, std, n in rmse_values]
        plt.errorbar(incoming_angle_values, [rmse for rmse, std, n in rmse_values], yerr=ci, fmt=".", capsize=3,
                     label=f"$a_s$ = {a_s:.5f}")

    plt.legend()
    plt.xlabel("Tx Bearing (\N{DEGREE SIGN})")
    plt.ylabel("RMSE (\N{DEGREE SIGN})")
    res_file.close()


def investigate_effect_of_bearing_parallel_helper(sectors, a_s, snr, angles):
    ant = sa.IdealSectoredAntenna(sectors, a_s=a_s)
    rmse_values = []
    for incoming_angle in angles:
        errs = []
        for repetition in range(NUM_REPETITIONS):
            sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, 0, 0, ant, 30, 30, incoming_angle)
            est_direction, _ = tsls.tsls_bearing_est(ant, sector_rx_powers, su.nat_to_db(2 * noise_var), L=4)
            est_direction = np.mod(est_direction, 360)
            errs.append(min(360 - np.abs(est_direction - incoming_angle), np.abs(est_direction - incoming_angle)))

        rmse = np.sqrt(np.mean(np.power(errs, 2)))
        std = np.std(errs)
        rmse_values.append((rmse, std, len(errs)))
    return a_s, rmse_values


def investigate_effect_of_bearing_parallel():
    """See what parameters have an effect on RMSE as a function of bearing."""
    incoming_angle_values = list(range(0, 360, 1))
    NUM_SECTORS = 6
    A_S_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6289062225842643]
    snr = 5

    res_file_name = datetime.now().strftime("results/drone_orientation/res_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,snr,num_repetitions,a_s,doa,rmse,std_dev\n")

    p = multi.Pool(NUM_PROC)
    args = []
    for a_s in A_S_VALUES:
        args.append((NUM_SECTORS, a_s, snr, incoming_angle_values))

    overall_results = p.starmap(investigate_effect_of_bearing_parallel_helper, args)

    for (a_s, results) in overall_results:
        for i in range(len(results)):
            rmse, std, num_repl = results[i]
            res_file.write(f"{NUM_SECTORS},{snr},{num_repl},{a_s},{incoming_angle_values[i]},{rmse},{std}\n")

    res_file.close()
    p.close()
    p.join()



def simulate_effect_of_drone_rotation_line_incremented():
    NUM_SECTORS = 4
    DRONE_SPACING = 10
    A_S = 0.4
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_line_incremented_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")

    res_file.write("num_sectors,a_s,drone_spacing,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [0, 1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                # plt.scatter([tx_loc_x], [tx_loc_y])

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + ((j * DRONE_SPACING) * np.sin(orth_dir))
                    loc_y = swarm_loc_y + ((j * DRONE_SPACING) * np.cos(orth_dir))
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=j * (360 / (NUM_SECTORS * num_drones)), ant=ant))

                    north_bearing = drones[-1].get_relative_bearing(0)
                    # plt.arrow(loc_x, loc_y, 5 * np.sin(np.deg2rad(north_bearing)), 5 * np.cos(np.deg2rad(north_bearing)),
                    #           color=PLOT_LINE_COLOURS[i], width=0.2)

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(f"{NUM_SECTORS},{A_S},{DRONE_SPACING},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt=".", capsize=3, label=f"{num_drones} drones [Incremented]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def simulate_effect_of_drone_rotation_line_uniform_0():
    NUM_SECTORS = 4
    DRONE_SPACING = 10
    A_S = 0.4
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_line_uniform0_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")

    res_file.write("num_sectors,a_s,drone_spacing,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [0, 1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                # plt.scatter([tx_loc_x], [tx_loc_y])

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + ((j * DRONE_SPACING) * np.sin(orth_dir))
                    loc_y = swarm_loc_y + ((j * DRONE_SPACING) * np.cos(orth_dir))
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=0, ant=ant))

                    # north_bearing = drones[-1].get_relative_bearing(0)
                    # plt.arrow(loc_x, loc_y, 5 * np.sin(np.deg2rad(north_bearing)), 5 * np.cos(np.deg2rad(north_bearing)),
                    #           color=PLOT_LINE_COLOURS[i], width=0.2)

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(f"{NUM_SECTORS},{A_S},{DRONE_SPACING},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt="x", capsize=3, label=f"{num_drones} drones [Uniform-0]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def simulate_effect_of_drone_rotation_line_partition():
    NUM_SECTORS = 4
    DRONE_SPACING = 10
    A_S = 0.4
    M_PART = 3
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_line_partition3_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,a_s,drone_spacing,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                # plt.scatter([tx_loc_x], [tx_loc_y])

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + ((j * DRONE_SPACING) * np.sin(orth_dir))
                    loc_y = swarm_loc_y + ((j * DRONE_SPACING) * np.cos(orth_dir))
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=(i % M_PART) * 360 / M_PART, ant=ant))

                    # north_bearing = drones[-1].get_relative_bearing(0)
                    # plt.arrow(loc_x, loc_y, 5 * np.sin(np.deg2rad(north_bearing)), 5 * np.cos(np.deg2rad(north_bearing)),
                    #           color=PLOT_LINE_COLOURS[i], width=0.2)

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(
                f"{NUM_SECTORS},{A_S},{DRONE_SPACING},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt="+", capsize=3, label=f"{num_drones} drones [{M_PART:.0f}-partition]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def simulate_effect_of_drone_rotation_line_random():
    NUM_SECTORS = 4
    DRONE_SPACING = 10
    A_S = 0.4
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_line_random_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")

    res_file.write("num_sectors,a_s,drone_spacing,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [0, 1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                # plt.scatter([tx_loc_x], [tx_loc_y])

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + ((j * DRONE_SPACING) * np.sin(orth_dir))
                    loc_y = swarm_loc_y + ((j * DRONE_SPACING) * np.cos(orth_dir))
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=np.random.randint(0, 360), ant=ant))

                    # north_bearing = drones[-1].get_relative_bearing(0)
                    # plt.arrow(loc_x, loc_y, 5 * np.sin(np.deg2rad(north_bearing)), 5 * np.cos(np.deg2rad(north_bearing)),
                    #           color=PLOT_LINE_COLOURS[i], width=0.2)

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(
                f"{NUM_SECTORS},{A_S},{DRONE_SPACING},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt="s", capsize=3, label=f"{num_drones} drones [Random]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def rotation_sim_lines():
    plt.figure()
    simulate_effect_of_drone_rotation_line_incremented()
    simulate_effect_of_drone_rotation_line_uniform_0()
    simulate_effect_of_drone_rotation_line_partition()
    simulate_effect_of_drone_rotation_line_random()
    plt.title("Drones in a line, drone line orthogonal to source")
    plt.xlabel('SNR (dB)')
    plt.ylabel("RMSE (\N{DEGREE SIGN})")


def simulate_effect_of_drone_rotation_circle_incremented():
    NUM_SECTORS = 4
    A_S = 0.4
    SWARM_RADIUS = 20
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_circle_incremented_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,a_s,radius,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [0, 1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)

                circle_x, circle_y, _ = pg.generate_points_uniform_on_circle_cart(num_drones, SWARM_RADIUS)

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + circle_x[j]
                    loc_y = swarm_loc_y + circle_y[j]
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=j * (360 / (NUM_SECTORS * num_drones)), ant=ant))

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(f"{NUM_SECTORS},{A_S},{SWARM_RADIUS},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt=".", capsize=3, label=f"{num_drones} drones [Incremented]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def simulate_effect_of_drone_rotation_circle_uniform_0():
    NUM_DRONES = 5
    NUM_SECTORS = 4
    SWARM_RADIUS = 20
    A_S = 0.4
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_circle_uniform0_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,a_s,radius,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [0, 1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                circle_x, circle_y, _ = pg.generate_points_uniform_on_circle_cart(num_drones, SWARM_RADIUS)

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + circle_x[j]
                    loc_y = swarm_loc_y + circle_y[j]
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=0.4)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=0, ant=ant))

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(f"{NUM_SECTORS},{A_S},{SWARM_RADIUS},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt="x", capsize=3, label=f"{num_drones} drones [Uniform-0]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def simulate_effect_of_drone_rotation_circle_partition():
    NUM_SECTORS = 4
    SWARM_RADIUS = 20
    A_S = 0.4
    M_PART = 3
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_circle_partition3_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,a_s,radius,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                circle_x, circle_y, _ = pg.generate_points_uniform_on_circle_cart(num_drones, SWARM_RADIUS)

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + circle_x[j]
                    loc_y = swarm_loc_y + circle_y[j]
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=(i % M_PART) * 360 / M_PART, ant=ant))

                    # north_bearing = drones[-1].get_relative_bearing(0)
                    # plt.arrow(loc_x, loc_y, 5 * np.sin(np.deg2rad(north_bearing)), 5 * np.cos(np.deg2rad(north_bearing)),
                    #           color=PLOT_LINE_COLOURS[i], width=0.2)

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(f"{NUM_SECTORS},{A_S},{SWARM_RADIUS},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt="+", capsize=3, label=f"{num_drones} drones [{M_PART:.0f}-partition]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def simulate_effect_of_drone_rotation_circle_random():
    NUM_SECTORS = 4
    A_S = 0.4
    SWARM_RADIUS = 20
    snr_values = list(np.arange(-1, 20, 1))

    res_file_name = datetime.now().strftime("results/drone_rot/res_circle_random_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("num_sectors,a_s,radius,num_repetitions,num_drones,snr,rmse,std_dev\n")

    for k in [0, 1, 2]:
        num_drones = [5, 9, 15][k]
        rmses = []
        rmse_cis = []
        for snr in snr_values:
            errs = []
            for repetition in rp.track(range(NUM_REPETITIONS)):
                # Create a drone array to pose
                swarm_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                swarm_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                tx_loc_x = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)
                tx_loc_y = np.random.uniform(-AREA_BOUND / 2, AREA_BOUND / 2)

                swarm_rx_tx_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                orth_dir = np.deg2rad(swarm_rx_tx_bearing + 90)

                circle_x, circle_y, _ = pg.generate_points_uniform_on_circle_cart(num_drones, SWARM_RADIUS)

                rx_drone_loc_x, rx_drone_loc_y = [], []
                drones = []
                for i in range(num_drones):
                    j = i - (num_drones // 2)
                    loc_x = swarm_loc_x + circle_x[j]
                    loc_y = swarm_loc_y + circle_y[j]
                    rx_drone_loc_x.append(loc_x)
                    rx_drone_loc_y.append(loc_y)

                    ant = sa.IdealSectoredAntenna(NUM_SECTORS, a_s=A_S)
                    drones.append(od.OrientableDrone(loc_x, loc_y, orientation=np.random.randint(0, 360), ant=ant))

                est_bearings = []
                for i in range(num_drones):
                    drone = drones[i]
                    true_bearing = pg.relative_bearing_2d(swarm_loc_x, swarm_loc_y, tx_loc_x, tx_loc_y)
                    sector_rx_powers, noise_var = generate_rx_powers_fixed(snr, drone.get_x_pos(), drone.get_y_pos(),
                                                                           drone.get_antenna(), tx_loc_x, tx_loc_y,
                                                                           drone.get_relative_bearing(true_bearing))
                    est_direction, _ = tsls.tsls_bearing_est(drone.get_antenna(), sector_rx_powers, su.nat_to_db(2 * noise_var),
                                                             L=3)
                    est_bearings.append(drone.get_true_bearing(np.mod(est_direction, 360)))

                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, fill=False)
                    # est_direction = drone.get_true_bearing(np.mod(est_direction, 360))
                    # plt.arrow(drone.get_x_pos(), drone.get_y_pos(), 5 * np.sin(np.deg2rad(est_direction)),
                    #           5 * np.cos(np.deg2rad(est_direction)), color=PLOT_LINE_COLOURS[i], width=0.2, hatch="x", fill=False)

                numerator, denominator = 0, 0
                for est in est_bearings:
                    numerator += np.sin(np.deg2rad(est))
                    denominator += np.cos(np.deg2rad(est))
                est_dir = np.mod(np.rad2deg(np.arctan2(numerator, denominator)), 360)
                err_fisher_and_lewis = min(360 - np.abs(est_dir - swarm_rx_tx_bearing), np.abs(est_dir - swarm_rx_tx_bearing))
                errs.append(err_fisher_and_lewis)

            rmse = np.sqrt(np.mean(np.power(errs, 2)))
            rmse_ci = 1.960 * np.std(errs) / np.sqrt(len(errs))
            print(f"{snr:.2f}: RMSE={rmse:.2f} \N{PLUS-MINUS SIGN} {rmse_ci:.2f} (p=0.05)")
            res_file.write(f"{NUM_SECTORS},{A_S},{SWARM_RADIUS},{len(errs)},{num_drones},{snr},{rmse},{np.std(errs)}\n")
            rmses.append(rmse)
            rmse_cis.append(rmse_ci)

        plt.errorbar(snr_values, rmses, yerr=rmse_cis, color=PLOT_LINE_COLOURS[k], fmt="s", capsize=3, label=f"{num_drones} drones [Random]")
    # plt.fill_between(snr_values, np.subtract(rmses, rmse_cis), np.add(rmses, rmse_cis))
    plt.legend()
    res_file.close()


def rotation_sim_circle():
    plt.figure()
    simulate_effect_of_drone_rotation_circle_incremented()
    simulate_effect_of_drone_rotation_circle_uniform_0()
    simulate_effect_of_drone_rotation_circle_partition()
    simulate_effect_of_drone_rotation_circle_random()
    plt.title("Drones in a circle")
    plt.xlabel('SNR (dB)')
    plt.ylabel("RMSE (\N{DEGREE SIGN})")


def main():
    os.environ["OMP_NUM_THREADS"] = f"{NUM_PROC}"
    simulate_rmse_based_on_snr_and_num_sectors()
    # simulate_performance_of_combining_estimates()
    # simulate_rmse_based_on_number_of_drones_fixed_snr()

    # investigate_effect_of_bearing_parallel()

    # rotation_sim_lines()
    # rotation_sim_circle()

    # plt.show()


if __name__ == "__main__":
    main()
