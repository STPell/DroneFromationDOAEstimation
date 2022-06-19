This repository contains the code used to generate the results included in 
S. Pell and A. Willig, “Using a drone formation with sectored antennas in
Search-And-Rescue: heuristics for orienting drones and moving the formation,”
in *2022 IEEE 33rd Annual International Symposium on Personal, Indoor and Mobile
Radio Communications (PIMRC) (IEEE PIMRC 2022)*, Sep. 2022.

# Setup Instructions

```commandline
pip3 install numpy matplotlib rich scipy
```

# Running Simulations

To recreate the simulations in the paper, you will need to modify the `main`
functions of some files to get them to run the simulation you wish to
replicate.

To replicate the simulations included in Figs. 1, 2, and 4 you will need to
modify [`main.py`](main.py) to enable the function which handles the simulation
you wish to replicate:
- Fig. 1 can be recreated by uncommenting `simulate_rmse_based_on_snr_and_num_sectors()`
- Fig. 2 can be recreated by uncommenting `investigate_effect_of_bearing_parallel()`
- Fig. 4 can be recreated by uncommenting `rotation_sim_lines()` and `rotation_sim_circle()`

Note for the best performance when simulating Fig. 2 you should make sure to
set `NUM_PROC` to the number of cores you want to use for the simulation.

To replicate the simulations included in Figs. 8 and 9 you will need to
modify [`EvaluateStepSizeEstimationMethods.py`](EvaluateStepSizeEstimationMethods.py).
- Fig. 8 can be recreated by uncommenting `simulate_angle_diff_to_distance_threshold_all_less_than_threshold()` and `simulate_angle_diff_to_distance_threshold_all_greater_than_threshold()`
- Fig. 9 can be recreated by uncommenting `step_counting_initial_mp()`

When performing the simulation necessary to recreate Fig. 9 you should make
sure to set `NUM_PROC` to the number of cores you want to use for the
simulation task.

# Recreating figures from the Paper

To recreate each figure from the paper, please see the
[readme in the results directory](results/README.md). If you wish to analyse
results from your simulation run you will have to modify the respective
Python scripts to pull from the results file you have generated.
