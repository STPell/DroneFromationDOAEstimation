"""
    Plot data exploring the effect of difference number of sectors on the RMSE of the DoA estimate.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   14/03/22
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink','tab:gray', 'tab:olive', 'tab:cyan']

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


def parse_args():
    parser = argparse.ArgumentParser(usage="%(prog)s...",
                                     description="Produce plots of the drone orientation data",
                                     add_help=True)
    parser.add_argument("-s", "--show", help="Show the plots before closing", action="store_true")
    parser.add_argument("-S", "--save", help="Save the plots before closing", action="store_true")
    parser.add_argument("-e", "--no-error-bars", help="Disable showing error bars on plots", action="store_true")
    parser.add_argument("-p", "--pgf", help="Save the plots before closing in the PGF format", action="store_true")
    parser.add_argument("-v", "--include_validation_extracted", help="Include the validation curve for M=6", action="store_true")
    parser.add_argument("-u", "--underlay_validation_graph", help="Show the screen-grab of the validation data under the plot", action="store_true")

    args = parser.parse_args()

    if args.show and args.pgf:
        raise ValueError("Cannot use PGF and show options at the same time")

    return args


def prepare_graph(args):
    if args.pgf:
        set_up_latex()

    # Taken from J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
    # Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286, Nov. 2015,
    # doi: 10.1109/JSAC.2015.2430292. Figure 6, TSLS + PW, L=3
    validation_x = [0.026, 0.162, 0.307, 0.461, 0.597, 0.742, 0.896, 1.051, 1.232, 1.404, 1.586, 1.767, 1.939, 2.148,
                    2.348, 2.583, 2.774, 3.001, 3.245, 3.499, 3.762, 3.989, 4.297, 4.597, 4.887, 5.186, 5.504, 5.812,
                    6.129, 6.465, 6.792, 7.127, 7.463, 7.798, 8.143, 8.515, 8.841, 9.213, 9.450, 9.939, 10.263, 10.609,
                    10.976, 11.330, 11.683, 12.029, 12.403, 12.763, 13.117, 13.477, 13.844, 14.204, 14.564, 14.917,
                    15.277, 15.644, 16.005, 16.371, 16.725, 17.085, 17.445, 17.805, 18.165, 18.525, 18.886, 19.253,
                    19.606, 19.986]
    validation_y = [8.606, 8.352, 8.125, 7.899, 7.654, 7.427, 7.191, 6.937, 6.738, 6.520, 6.293, 6.076, 5.858, 5.649,
                    5.441, 5.232, 5.042, 4.833, 4.643, 4.470, 4.271, 4.098, 3.944, 3.817, 3.672, 3.536, 3.400, 3.273,
                    3.146, 3.028, 2.929, 2.829, 2.747, 2.665, 2.593, 2.539, 2.484, 2.430, 2.357, 2.303, 2.277, 2.243,
                    2.209, 2.182, 2.141, 2.107, 2.080, 2.060, 2.046, 2.019, 1.998, 1.978, 1.964, 1.951, 1.937, 1.930,
                    1.917, 1.896, 1.896, 1.869, 1.876, 1.869, 1.862, 1.856, 1.849, 1.849, 1.842, 1.835]

    fil = open("effect_of_sectors/floating_signal_strength_res_20220420-113331.csv")
    lines = fil.readlines()[1:]  # Ignore first line as its a header
    fil.close()

    fig, ax = plt.subplots()

    if args.underlay_validation_graph:
        img = plt.imread("effect_of_sectors/validation_graph.png")
        ax.imshow(img, extent=[0, 20, 0, 12])

    num_sectorss = []
    snrs = []
    rmses = []
    cis = []
    # num_sectors, snr, num_repetitions, a_s, rmse, std_dev
    for l in lines:
        num_sectors, snr, num_repl, a_s, rmse, std_dev = l.split(',')
        num_sectorss.append(int(num_sectors))
        snrs.append(float(snr))
        rmses.append(float(rmse))
        cis.append(1.961 * float(std_dev) / np.sqrt(int(num_repl)))

    num_sectorss_set = set(num_sectorss)
    for num_sectors in sorted(list(num_sectorss_set)):
        snr_values = [snrs[i] for i in range(len(num_sectorss)) if num_sectorss[i] == num_sectors]
        rms_values = [rmses[i] for i in range(len(num_sectorss)) if num_sectorss[i] == num_sectors]
        ci_values = [cis[i] for i in range(len(num_sectorss)) if num_sectorss[i] == num_sectors]
        if args.no_error_bars:
            ax.plot(snr_values, rms_values, label=f"$M$ = {num_sectors}", marker=".", linestyle="None")
        else:
            ax.errorbar(snr_values, rms_values, yerr=ci_values, label=f"$M$ = {num_sectors}", marker=".",
                         linestyle="None")

    if args.include_validation_extracted:
        ax.plot(validation_x, validation_y, label="Validation Data (TSLS + PW, $L$=3)")

    plt.xlim([-1, 19])
    plt.ylim([0, 11])

    lgd = plt.legend()
    if args.show:
        plt.xlabel("SNR (dB)")
        plt.ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")
        plt.show()

    if args.pgf:
        plt.xlabel("SNR (dB)")
        plt.ylabel("RMSE $\widehat{\phi{}}$ ($^\circ$)")
        plt.savefig("rmse_as_sectors_increases.pgf", bbox_extra_artists=(lgd,))

    if args.save:
        plt.xlabel("SNR (dB)")
        plt.ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")
        plt.savefig("rmse_as_sectors_increases.png", bbox_extra_artists=(lgd,))


def main():
    args = parse_args()
    prepare_graph(args)


if __name__ == "__main__":
    main()
