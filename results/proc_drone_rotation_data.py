import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

# Import stuff for inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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


def process_circle(args):
    fig, ax = plt.subplots()
    ax_ins = inset_axes(ax, width="50%", height="45%", loc="center right")

    for filename, desc in [("res_circle_partition3_20220223-041009.csv", "Partition, $G=3$"),
                           ("res_circle_incremented_20220223-012813.csv", "Incremented"),
                           ("res_circle_uniform0_20220223-024854.csv", "Uniform"),
                           ("res_circle_random_20220223-051709.csv", "Random")]:
        fil = open(f"drone_rot/{filename}")
        lines = fil.readlines()[1:]  # Ignore first line as its a header
        fil.close()

        num_drones = []
        snrs = []
        rmses = []
        cis = []

        # num_sectors,a_s,radius,num_repetitions,num_drones,snr,rmse,std_dev
        for l in lines:
            num_sectors, a_s, radius, num_repl, D, snr, rmse, std_dev = l.split(',')
            num_drones.append(int(D))
            snrs.append(float(snr))
            rmses.append(float(rmse))
            cis.append(1.961 * float(std_dev) / np.sqrt(int(num_repl)))

        # for D in sorted(list(set(num_drones))):
        for D in [9]:
            snr_values = [snrs[i] for i in range(len(num_drones)) if num_drones[i] == D]
            rms_values = [rmses[i] for i in range(len(num_drones)) if num_drones[i] == D]
            ci_values = [cis[i] for i in range(len(num_drones)) if num_drones[i] == D]
            ax.errorbar(snr_values, rms_values, yerr=ci_values, label=f"{desc}", marker=".", linestyle="None")
            ax_ins.errorbar(snr_values, rms_values, yerr=ci_values, label=f"{desc}", marker=".", linestyle="None")

    ax.set_xlim([-1.5, 19])
    ax.set_ylim([0, 4.2])
    
    mark_inset(ax, ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")
        
    ax_ins.set_xlim([10.5, 18.5])
    ax_ins.set_ylim([0.625, 0.85])
    
    lgd = ax.legend()
    if args.show:
        ax.set_title("Circle")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")

    if args.pgf:
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE $\widehat{\phi{}}$ ($^\circ$)")
        plt.savefig("rmse_drone_rot_comp_circle.pgf", bbox_extra_artists=(lgd,))

    if args.save:
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")
        plt.savefig("rmse_drone_rot_comp_circle.png", bbox_extra_artists=(lgd,))
 
 
def process_line(args):
    fig, ax = plt.subplots()
    ax_ins = inset_axes(ax, width="50%", height="45%", loc="center right")

    for filename, desc in [("res_line_partition3_20220222-175351.csv", "Partition, $G=3$"),
                           ("res_line_incremented_20220222-151451.csv", "Incremented"),
                           ("res_line_uniform0_20220222-163423.csv", "Uniform"),
                           ("res_line_random_20220222-190010.csv", "Random")]:
        fil = open(f"drone_rot/{filename}")
        lines = fil.readlines()[1:]  # Ignore first line as its a header
        fil.close()

        num_drones = []
        snrs = []
        rmses = []
        cis = []

        # num_sectors,a_s,radius,num_repetitions,num_drones,snr,rmse,std_dev
        for l in lines:
            num_sectors, a_s, radius, num_repl, D, snr, rmse, std_dev = l.split(',')
            num_drones.append(int(D))
            snrs.append(float(snr))
            rmses.append(float(rmse))
            cis.append(1.961 * float(std_dev) / np.sqrt(int(num_repl)))

        # for D in sorted(list(set(num_drones))):
        for D in [9]:
            snr_values = [snrs[i] for i in range(len(num_drones)) if num_drones[i] == D]
            rms_values = [rmses[i] for i in range(len(num_drones)) if num_drones[i] == D]
            ci_values = [cis[i] for i in range(len(num_drones)) if num_drones[i] == D]
            ax.errorbar(snr_values, rms_values, yerr=ci_values, label=f"{desc}", marker=".", linestyle="None")
            ax_ins.errorbar(snr_values, rms_values, yerr=ci_values, label=f"{desc}", marker=".", linestyle="None")

    ax.set_xlim([-1.5, 19])
    ax.set_ylim([0, 4.2])
    
    mark_inset(ax, ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")
        
    ax_ins.set_xlim([10.5, 18.5])
    ax_ins.set_ylim([0.625, 0.85])
    
    lgd = ax.legend()
    if args.show:
        ax.set_title("Line")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")

    if args.pgf:
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE $\widehat{\phi{}}$ ($^\circ$)")
        fig.savefig("rmse_drone_rot_comp_line.pgf", bbox_extra_artists=(lgd,))

    if args.save:
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE $\widehat{\phi{}}$ (\N{DEGREE SIGN})")
        fig.savefig("rmse_drone_rot_comp_line.png", bbox_extra_artists=(lgd,))


def parse_args():
    parser = argparse.ArgumentParser(usage="%(prog)s...",
                                     description="Produce plots of the drone rotation data",
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

    process_circle(args)
    process_line(args)
    #disk(args)
    
    if args.show:
        plt.show()
    

if __name__ == "__main__":
    main()
