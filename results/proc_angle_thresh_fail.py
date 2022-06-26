import string
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import itertools

COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


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


def process_thresh_false_positive(args):
    f = open("angle_thresh_fail/angle_thresh_fail_long", "r")
    f.readline()  # First line is the header, ignore it
    res = []

    line = f.readline()
    while line != "":
        line = line.strip()
        # SNR: 0, Dist Threshold: 100 m, TP: 98280, FP: 61, TN: 4, FN: 1655
        snr_s, thresh_s, tp_s, fp_s, tn_s, fn_s = line.strip().split(",")
        res.append((float(snr_s.split(":")[1].strip()), int(thresh_s.split(":")[1].strip().split(" ")[0].strip()),
                    int(tn_s.split(":")[1].strip()), int(fp_s.split(":")[1].strip())))
        line = f.readline()

    f.close()

    thresholds = list(set([thresh for snr, thresh, tn, fp in res]))

    plt.figure()
    for threshold in thresholds:
        snrs = [snr for snr, thresh, tp, fp in res if thresh == threshold]
        fp_rate = [100 * float(fp) / 100000 for snr, thresh, tp, fp in res if thresh == threshold]
        if args.pgf:
            plt.plot(snrs, fp_rate, '-o', label=f'Threshold = {threshold:.0f}\,m')
        else:
            plt.plot(snrs, fp_rate, '-o', label=f'Threshold = {threshold:.0f} m')

    lgd = plt.legend(loc='upper right')
    plt.xlim([0, 30])
    plt.ylim([0, 90])
    plt.xlabel("SNR (dB)")
    plt.ylabel("False Positive Rate (\%)")

    if args.pgf:
        plt.savefig(f"long_distance_dist_est.pgf", bbox_extra_artists=(lgd,), bbox_inches="tight")

    if args.save:
        plt.savefig(f"long_distance_dist_est.png", bbox_extra_artists=(lgd,))
    

def process_thresh_false_negative(args):
    f = open("angle_thresh_fail/angle_thresh_fail_short", "r")
    f.readline()  # First line is the header, ignore it
    res = []

    line = f.readline()
    while line != "":
        line = line.strip()
        # SNR: 0, Dist Threshold: 100 m, TP: 98280, FP: 61, TN: 4, FN: 1655
        snr_s, thresh_s, tp_s, fp_s, tn_s, fn_s = line.strip().split(",")
        res.append((float(snr_s.split(":")[1].strip()), int(thresh_s.split(":")[1].strip().split(" ")[0].strip()),
                    int(tp_s.split(":")[1].strip()), int(fn_s.split(":")[1].strip())))
        line = f.readline()

    f.close()

    thresholds = list(set([thresh for snr, thresh, tp, fn in res]))

    plt.figure()
    for threshold in thresholds:
        snrs = [snr for snr, thresh, tp, fn in res if thresh == threshold]
        fn_rate = [100 * float(fn) / 100000 for snr, thresh, tp, fn in res if thresh == threshold]
        if args.pgf:
            plt.plot(snrs, fn_rate, '-o', label=f'Threshold = {threshold:.0f}\,m')
        else:
            plt.plot(snrs, fn_rate, '-o', label=f'Threshold = {threshold:.0f} m')

    lgd = plt.legend(loc='upper right')
    plt.xlim([0, 30])
    plt.ylim([0, 12])
    plt.xlabel("SNR (dB)")
    plt.ylabel("False Negative Rate (\%)")

    if args.pgf:
        plt.savefig(f"shirt_distance_dist_est.pgf", bbox_extra_artists=(lgd,), bbox_inches="tight")

    if args.save:
        plt.savefig(f"shirt_distance_dist_est.png", bbox_extra_artists=(lgd,))



def parse_args():
    parser = argparse.ArgumentParser(usage="%(prog)s ...",
                                     description="Produce plots of the drone orientation data",
                                     add_help=True)
    parser.add_argument("-s", "--show", help="Show the plots before closing", action="store_true")
    parser.add_argument("-S", "--save", help="Save the plots before closing", action="store_true")
    parser.add_argument("-p", "--pgf", help="Save the plots before closing in the PGF format", action="store_true")

    args = parser.parse_args()

    if args.show and args.pgf:
        raise ValueError("Cannot use PGF and show options at the same time")

    return args


def main():
    args = parse_args()
    if args.pgf:
        set_up_latex()

    process_thresh_false_negative(args)
    process_thresh_false_positive(args)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
