import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

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

    args = parser.parse_args()

    if args.show and args.pgf:
        raise ValueError("Cannot use PGF and show options at the same time")

    return args


def main():
    args = parse_args()

    if args.pgf:
        set_up_latex()

    fil = open("drone_orientation/res_20220223-151002.csv")
    lines = fil.readlines()[1:]  # Ignore first line as its a header
    fil.close()

    num_sectorss = None
    snrs = None,
    a_ss = []
    doas = []
    rmses = []
    cis = []
    # num_sectors, snr, num_repetitions, a_s, doa, rmse, std_dev
    for l in lines:
        num_sectors, snr, num_repl, a_s, doa, rmse, std_dev = l.split(',')
        num_sectorss = int(num_sectors)
        snrs = float(snr)
        a_ss.append(float(a_s))
        doas.append(float(doa))
        rmses.append(float(rmse))
        cis.append(1.961 * float(std_dev) / np.sqrt(int(num_repl)))

    a_s_set = set(a_ss)
    for a_s in sorted(list(a_s_set))[:-1]:
    # for a_s in sorted(list(a_s_set)):
        doa_values = [doas[i] for i in range(len(a_ss)) if a_ss[i] == a_s]
        rms_values = [rmses[i] for i in range(len(a_ss)) if a_ss[i] == a_s]
        ci_values = [cis[i] for i in range(len(a_ss)) if a_ss[i] == a_s]
        if args.no_error_bars:
            plt.plot(doa_values, rms_values, label=f"$a_s$ = {a_s}", marker=".", linestyle="None")
        else:
            plt.errorbar(doa_values, rms_values, yerr=ci_values, label=f"$a_s$ = {a_s}", marker=".", linestyle="None")

    plt.xlim([0, 360])
    plt.ylim([0, 40])
    lgd = plt.legend()
    if args.show:
        plt.xlabel("Transmitter DoA $\phi{}$ (\N{DEGREE SIGN})")
        plt.ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")
        plt.show()

    if args.pgf:
        plt.xlabel("Transmitter DoA $\phi{}$ ($^\circ$)")
        plt.ylabel("RMSE $\widehat{\phi{}}$ ($^\circ$)")
        plt.savefig("rmse_est_doa_per_doa.pgf", bbox_extra_artists=(lgd,))

    if args.save:
        plt.xlabel("Transmitter DoA $\phi{}$ (\N{DEGREE SIGN})")
        plt.ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")
        plt.savefig("rmse_est_doa_per_doa.png", bbox_extra_artists=(lgd,))


if __name__ == "__main__":
    main()
