"""
    Perform simulation exploring delta angle thresholding to categorise the distance to the transmitter

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   06/04/22

    Changelog:
        26/04/22 Time how long it takes for the formation to get within x metres of the Tx
        04/05/22 Parallelise timing code to speed up simulation time
"""

import DroneSwarm as ds
import matplotlib.pyplot as plt
import numpy as np
import angle_per_distance as ad
from datetime import datetime
import multiprocessing as mp

NUM_REPETITIONS = int(1e3)
AREA_SIZE = 10000 * np.sqrt(2) / 2
SNR = 5
MAX_STEPS = 1000
NUM_PROC = 45


PLOT_LINE_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:gray', 'tab:olive', 'tab:cyan']

PLOTS = {'ecdf': 1}


def generate_random_loc_in_area(side_x, side_y):
    x = np.random.uniform(-side_x / 2, side_x / 2)
    y = np.random.uniform(-side_y / 2, side_y / 2)
    return x, y


def bearing_diff(b1, b2):
    return min(360 - np.abs(b2 - b1), np.abs(b2 - b1))


def simulate_angle_diff_to_distance_threshold_all_less_than_threshold():
    DIST_THRESHOLDS = [100, 500, 1000]
    REPLICATIONS = 100000
    SNRS = range(0, 31, 2)

    plt.figure()

    print("All distances less than thresholds:")
    for dist_threshold in DIST_THRESHOLDS:
        area_size = dist_threshold * np.sqrt(2) / 2
        fn_per_angle = []
        fn_rate = []
        for snr in SNRS:
            TP, TN, FP, FN = 0, 0, 0, 0
            for repl in range(REPLICATIONS):
                tx_loc_x, tx_loc_y = generate_random_loc_in_area(area_size, area_size)
                swarm_start_x, swarm_start_y = generate_random_loc_in_area(area_size, area_size)
                left_swarm = ds.CircularDroneFormation(swarm_start_x - 20, swarm_start_y, 10, 5, 6, 0.4)
                right_swarm = ds.CircularDroneFormation(swarm_start_x + 20, swarm_start_y, 10, 5, 6, 0.4)
                left_swarm.set_tx_loc(tx_loc_x, tx_loc_y)
                right_swarm.set_tx_loc(tx_loc_x, tx_loc_y)

                # left_swarm.use_incremented_drone_orientation()
                # right_swarm.use_incremented_drone_orientation()

                est_bearing_left = left_swarm.estimate_tx_bearing(SNR)
                est_bearing_right = right_swarm.estimate_tx_bearing(SNR)
                angle_delta = np.abs(est_bearing_left - est_bearing_right)

                left_x, left_y = left_swarm.get_formation_centre()
                real_dist = np.linalg.norm(np.subtract([tx_loc_x, tx_loc_y], [left_x, left_y]))

                if angle_delta > ad.est_angle_difference(40, est_bearing_left, dist_threshold):
                    if real_dist < dist_threshold:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if dist_threshold <= real_dist:
                        TN += 1
                    else:
                        FN += 1
                        fn_per_angle.append(est_bearing_left)
            # plt.figure()
            # plt.title(f"False Negative - $d$ < {dist_threshold}")
            # plt.hist(fn_per_angle, bins=[i for i in range(0, 362, 5)], density=True)
            # plt.xlabel('Bearing (\N{DEGREE SIGN})')
            # plt.xlim([0, 360])
            print(f"\tSNR: {snr}, Dist Threshold: {dist_threshold} m, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
            fn_rate.append(100 * FN / REPLICATIONS)
        plt.plot(SNRS, fn_rate, "-o", label=f"Threshold = {dist_threshold} m")
    plt.legend()

    plt.xlim([0, 30])
    plt.ylim([0, 12.5])
    plt.xlabel("SNR (dB)")
    plt.ylabel("False Negative Rate (\%)")
    plt.savefig("shirt_distance_dist_est.pgf")


def simulate_angle_diff_to_distance_threshold_all_greater_than_threshold():
    DIST_THRESHOLDS = [100, 500, 1000]
    REPLICATIONS = 100000
    SNRS = range(0, 31, 2)

    plt.figure()

    print("All distances more than thresholds:")
    for dist_threshold in DIST_THRESHOLDS:
        area_size = 1e4
        fp_per_angle = []
        fp_rate = []
        for snr in SNRS:
            TP, TN, FP, FN = 0, 0, 0, 0
            for repl in range(REPLICATIONS):
                tx_loc_x, tx_loc_y = generate_random_loc_in_area(area_size, area_size)
                swarm_start_x, swarm_start_y = generate_random_loc_in_area(area_size, area_size)
                while np.linalg.norm(np.subtract([tx_loc_x, tx_loc_y], [swarm_start_x, swarm_start_y])) < dist_threshold + 100:
                    swarm_start_x, swarm_start_y = generate_random_loc_in_area(area_size, area_size)

                left_swarm = ds.CircularDroneFormation(swarm_start_x - 20, swarm_start_y, 10, 5, 6, 0.4)
                right_swarm = ds.CircularDroneFormation(swarm_start_x + 20, swarm_start_y, 10, 5, 6, 0.4)
                left_swarm.set_tx_loc(tx_loc_x, tx_loc_y)
                right_swarm.set_tx_loc(tx_loc_x, tx_loc_y)

                # left_swarm.use_incremented_drone_orientation()
                # right_swarm.use_incremented_drone_orientation()

                est_bearing_left = left_swarm.estimate_tx_bearing(snr)
                est_bearing_right = right_swarm.estimate_tx_bearing(snr)
                angle_delta = np.abs(est_bearing_left - est_bearing_right)
                # TODO: If the dumbbell is not parallel to the x-axis, you may need to figure out how to correct for the
                #       distortion introduced by the angle between the two drones into the system (assuming that the drones
                #       maintain their orientation to due north). If the drones change their orientation to rotate together,
                #       such that their local coordinate systems aren't oriented with due north at 0 degrees, things will
                #       also change. However, for now I'm going to ignore that.

                left_x, left_y = left_swarm.get_formation_centre()
                real_dist = np.linalg.norm(np.subtract([tx_loc_x, tx_loc_y], [left_x, left_y]))

                if angle_delta > ad.est_angle_difference(20, est_bearing_left, dist_threshold):
                    if real_dist < dist_threshold:
                        TP += 1
                    else:
                        FP += 1
                        fp_per_angle.append(est_bearing_left)
                else:
                    if dist_threshold <= real_dist:
                        TN += 1
                    else:
                        FN += 1
            # plt.figure()
            # plt.title(f"False Positive - $d$ > {dist_threshold}")
            # plt.hist(fp_per_angle, bins=[i for i in range(0, 362, 5)], density=True)
            # plt.xlabel('Bearing (\N{DEGREE SIGN})')
            # plt.xlim([0, 360])
            print(f"\tSNR: {snr}, Dist Threshold: {dist_threshold} m, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
            fp_rate.append(100 * FP / REPLICATIONS)
        plt.plot(SNRS, fp_rate, "-o", label=f"Threshold = {dist_threshold} m")

    plt.legend()
    plt.xlim([0, 30])
    plt.ylim([0, 90])
    plt.xlabel("SNR (dB)")
    plt.ylabel("False Positive Rate (\%)")
    plt.savefig("long_distance_dist_est.pgf")


def step_counting_single_repl(tx_range, dist_thresholds):
    DRONE_SPEED = 5  # m/s
    MEASUREMENT_TIME = 10  # s
    MAX_STEPS = 200
    FINISH_DISTANCE = 10

    tx_loc_x, tx_loc_y = (0, 0)
    repl_time = 0
    bearing = np.deg2rad(np.random.uniform(0, 360))
    form_centre_x, form_centre_y = tx_range * np.sin(bearing), tx_range * np.cos(bearing)

    left_swarm = ds.CircularDroneFormation(form_centre_x - 20, form_centre_y, 10, 5, 6, 0.4)
    right_swarm = ds.CircularDroneFormation(form_centre_x + 20, form_centre_y, 10, 5, 6, 0.4)
    left_swarm.set_tx_loc(tx_loc_x, tx_loc_y)
    right_swarm.set_tx_loc(tx_loc_x, tx_loc_y)

    left_swarm.use_incremented_drone_orientation()
    right_swarm.use_incremented_drone_orientation()

    real_dist = tx_range

    step_count = 0
    step_history = []
    while step_count < MAX_STEPS and real_dist > FINISH_DISTANCE:
        step_count += 1
        est_bearing_left = left_swarm.estimate_tx_bearing()
        est_bearing_right = right_swarm.estimate_tx_bearing()
        angle_delta = np.abs(est_bearing_left - est_bearing_right)

        overall_est_bearing = left_swarm._combine_estimate_bearings(left_swarm.get_bearing_last_estimates() + right_swarm.get_bearing_last_estimates())

        step_size = dist_thresholds[0]
        for thresh in dist_thresholds:
            if angle_delta > ad.est_angle_difference(40, est_bearing_left, thresh):
                step_size = thresh

        repl_time += MEASUREMENT_TIME + (step_size / DRONE_SPEED)
        step_history.append(step_size)
        left_swarm.move_swarm(overall_est_bearing, step_size)
        right_swarm.move_swarm(overall_est_bearing, step_size)

        left_x, left_y = left_swarm.get_formation_centre()
        right_x, right_y = right_swarm.get_formation_centre()
        real_dist = np.linalg.norm(np.subtract([tx_loc_x, tx_loc_y], [(left_x + right_x) / 2, (left_y + right_y) / 2]))
    return repl_time, step_count, real_dist


def step_counting_initial_mp():
    DIST_THRESHOLDS_ARRAY = [[100, 50, 5], [200, 100, 50, 5], [500, 200, 100, 50, 5], [1000, 500, 200, 100, 50, 5]]
    REPLICATIONS = 100000
    DRONE_SPEED = 5  # m/s
    MEASUREMENT_TIME = 10  # s
    RANGES = [1628, 3628]  # m
    FINISH_DISTANCE = 10

    res_file_name = datetime.now().strftime("results/angle_thresh_timing/res_thresh_timing_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("drone_speed,measurement_time,finish_dist,tx_range,dist_thresholds,repl_no,time_taken,num_steps,final_dist\n")

    for tx_range in RANGES:
        for dist_thresholds in DIST_THRESHOLDS_ARRAY:
            threshold_list = ";".join([str(i) for i in dist_thresholds])
            log_file_line_prefix = f"{DRONE_SPEED},{MEASUREMENT_TIME},{FINISH_DISTANCE},{tx_range},{threshold_list}"

            thread_pool = mp.Pool(NUM_PROC)
            tasks = [(tx_range, dist_thresholds) for i in range(REPLICATIONS)]
            res = thread_pool.starmap(step_counting_single_repl, tasks)
            thread_pool.close()
            thread_pool.join()

            for i in range(REPLICATIONS):
                repl_time, step_count, real_dist = res[i]
                res_file.write(f"{log_file_line_prefix},{i},{repl_time},{step_count},{real_dist}\n")
    res_file.close()


def step_counting_initial():
    DIST_THRESHOLDS_ARRAY = [[100, 50, 5], [200, 100, 50, 5], [500, 200, 100, 50, 5], [1000, 500, 200, 100, 50, 5]]
    REPLICATIONS = 10000
    DRONE_SPEED = 5  # m/s
    MEASUREMENT_TIME = 10  # s
    RANGES = [1628, 3628]  # m
    MAX_STEPS = 200
    FINISH_DISTANCE = 10

    tx_loc_x, tx_loc_y = (0, 0)

    res_file_name = datetime.now().strftime("results/angle_thresh_timing/res_thresh_timing_%Y%m%d-%H%M%S.csv")
    res_file = open(res_file_name, "w")
    res_file.write("drone_speed,measurement_time,finish_dist,tx_range,dist_thresholds,repl_no,time_taken,num_steps,final_dist\n")

    for tx_range in RANGES:
        for dist_thresholds in DIST_THRESHOLDS_ARRAY:
            threshold_list = ";".join([str(i) for i in dist_thresholds])
            log_file_line_prefix = f"{DRONE_SPEED},{MEASUREMENT_TIME},{FINISH_DISTANCE},{tx_range},{threshold_list}"

            time_taken = []
            for repl in range(REPLICATIONS):
                repl_time = 0
                bearing = np.deg2rad(np.random.uniform(0, 360))
                form_centre_x, form_centre_y = tx_range * np.sin(bearing), tx_range * np.cos(bearing)

                left_swarm = ds.CircularDroneFormation(form_centre_x - 20, form_centre_y, 10, 5, 6, 0.4)
                right_swarm = ds.CircularDroneFormation(form_centre_x + 20, form_centre_y, 10, 5, 6, 0.4)
                left_swarm.set_tx_loc(tx_loc_x, tx_loc_y)
                right_swarm.set_tx_loc(tx_loc_x, tx_loc_y)

                left_swarm.use_incremented_drone_orientation()
                right_swarm.use_incremented_drone_orientation()

                real_dist = tx_range

                step_count = 0
                step_history = []
                while step_count < MAX_STEPS and real_dist > FINISH_DISTANCE:
                    step_count += 1
                    est_bearing_left = left_swarm.estimate_tx_bearing()
                    est_bearing_right = right_swarm.estimate_tx_bearing()
                    angle_delta = np.abs(est_bearing_left - est_bearing_right)

                    overall_est_bearing = left_swarm._combine_estimate_bearings(left_swarm.get_bearing_last_estimates() + right_swarm.get_bearing_last_estimates())

                    step_size = dist_thresholds[0]
                    for thresh in dist_thresholds:
                        if angle_delta > ad.est_angle_difference(40, est_bearing_left, thresh):
                            step_size = thresh

                    repl_time += MEASUREMENT_TIME + (step_size / DRONE_SPEED)
                    step_history.append(step_size)
                    left_swarm.move_swarm(overall_est_bearing, step_size)
                    right_swarm.move_swarm(overall_est_bearing, step_size)

                    left_x, left_y = left_swarm.get_formation_centre()
                    right_x, right_y = right_swarm.get_formation_centre()
                    real_dist = np.linalg.norm(np.subtract([tx_loc_x, tx_loc_y], [(left_x + right_x) / 2, (left_y + right_y) / 2]))
                time_taken.append((repl_time, step_count, real_dist))
                res_file.write(f"{log_file_line_prefix},{repl},{repl_time},{step_count},{real_dist}\n")

            print(tx_range, dist_thresholds, np.average([repl_time for repl_time, step_count, real_dist in time_taken]), np.std([repl_time for repl_time, step_count, real_dist in time_taken]))
    res_file.close()


def main():
    # simulate_angle_diff_to_distance_threshold_all_less_than_threshold()
    # simulate_angle_diff_to_distance_threshold_all_greater_than_threshold()
    # step_counting_initial()
    step_counting_initial_mp()


if __name__ == "__main__":
    main()
