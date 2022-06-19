import string
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import itertools
import samUtilities as su
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink','tab:gray', 'tab:olive', 'tab:cyan']


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


def custom_mark_inset(axA, axB, fc='None', ec='k'):
    """Taken from https://stackoverflow.com/a/45078051 and modified"""
    xx = axB.get_xlim()
    yy = axB.get_ylim()

    xy = (xx[0], yy[0])
    width = xx[1] - xx[0]
    height = yy[1] - yy[0]

    pp = axA.add_patch(patches.Rectangle(xy, width, height, fc=fc, ec=ec))

    p1 = axA.add_patch(patches.ConnectionPatch(
        xyA=(xx[0] + width, yy[0] + height), xyB=(xx[0], yy[1]),
        coordsA='data', coordsB='data',
        axesA=axA, axesB=axB, ec=ec))

    p2 = axA.add_patch(patches.ConnectionPatch(
        xyA=(xx[1], yy[0]), xyB=(xx[1], yy[1]),
        coordsA='data', coordsB='data',
        axesA=axA, axesB=axB, ec=ec))

    return pp, p1, p2


def process_thresh_per_range_ecdf(args):
    f = open("angle_thresh_timing/res_thresh_timing_20220504-073437.csv", "r")
    f.readline()  # First line is the header, ignore it
    res = []

    line = f.readline()
    while line != "":
        line = line.strip()
        # drone_speed,measurement_time,finish_dist,tx_range,dist_thresholds,repl_no,time_taken,num_steps,final_dist
        _, _, _, tx_range, thresholds, _, time, _, final_dist = line.split(",")
        thresh_list = [float(t) for t in thresholds.split(";")]
        res.append((float(tx_range), thresh_list, float(time), float(final_dist)))
        line = f.readline()

    f.close()

    unique_tx_ranges = sorted(list(set([tx_range for tx_range, _, _, _ in res])))
    unique_thresholds = list(thresh for thresh, _ in itertools.groupby(sorted([thresh_list for _, thresh_list, _, _ in res])))

    for range_idx in range(len(unique_tx_ranges)):
        averages = []
        cis = []

        tx_range = unique_tx_ranges[range_idx]
        fig, fig_ax = plt.subplots()

        if args.inset:
            ax_ins = inset_axes(fig_ax, width="55%", height="55%", loc="upper right", borderpad=1.75)
        
        filtered_res = [(thresh, time, final_dist) for tx_range_, thresh, time, final_dist in res if tx_range_ == tx_range]
        min_time = 1e8
        max_time = -1e8
        for i in range(len(unique_thresholds)):
            thresholds = unique_thresholds[i]
            time = [t for thresh, t, d in filtered_res if thresholds == thresh and d < 10]
            av = np.average(time)
            ci = 1.960 * np.std(time) / np.sqrt(len(time))
            print(f"{tx_range}, {thresholds}: {av} \N{PLUS-MINUS SIGN} {ci}")
            averages.append(av)
            cis.append(ci)

            if len(time) > 0:
                min_time = min(min_time, min(time))
                max_time = max(max_time, max(time))

                if args.pgf:
                    label = ", ".join([f"{t:.0f}\,m" for t in thresholds])
                else:
                    label = ", ".join([f"{t:.0f} m" for t in thresholds])

                su.plot_ecdf_axes(fig_ax, time, label=label)
                if args.inset:
                    su.plot_ecdf_axes(ax_ins, time, label=label)

        fig_ax.set_xlabel("Time (s)")
        lgd = fig_ax.legend()
        fig_ax.set_ylim([0, 1.02])
        fig_ax.set_xlim([(min_time // 100) * 100, (max_time // 100) * 100])

        if args.inset:
            ax_ins.set_xlim([(min_time // 100) * 100, ((max(averages) + max(cis))// 100) * 100 + 500])
            ax_ins.set_ylim([0, 1.0])
            if args.mark_insets:
                custom_mark_inset(fig_ax, ax_ins, ec='0.8')

        if args.show:
            fig.suptitle(f"$R={tx_range}$")

        if args.pgf:
            plt.savefig(f"time_per_steps_range_{tx_range:.0f}.pgf", bbox_extra_artists=(lgd,))

        if args.save:
            plt.savefig(f"time_per_steps_range_{tx_range:.0f}.png", bbox_extra_artists=(lgd,))


def process_thresh_per_range_box_plot(args):
    f = open("angle_thresh_timing/res_thresh_timing_20220503-074224.csv", "r")
    f.readline()  # First line is the header, ignore it
    res = []

    line = f.readline()
    while line != "":
        line = line.strip()
        # drone_speed,measurement_time,finish_dist,tx_range,dist_thresholds,repl_no,time_taken,num_steps,final_dist
        _, _, _, tx_range, thresholds, _, time, _, final_dist = line.split(",")
        thresh_list = [float(t) for t in thresholds.split(";")]
        res.append((float(tx_range), thresh_list, float(time), float(final_dist)))
        line = f.readline()

    f.close()

    unique_tx_ranges = sorted(list(set([tx_range for tx_range, _, _, _ in res])))
    unique_thresholds = list(thresh for thresh, _ in itertools.groupby(sorted([thresh_list for _, thresh_list, _, _ in res])))

    averages = []
    cis = []

    for range_idx in range(len(unique_tx_ranges)):
        tx_range = unique_tx_ranges[range_idx]
        fig, fig_ax = plt.subplots()
        filtered_res = [(thresh, time, final_dist) for tx_range_, thresh, time, final_dist in res if tx_range_ == tx_range]
        overall_times = []
        for i in range(len(unique_thresholds)):
            thresholds = unique_thresholds[i]
            time = [t for thresh, t, d in filtered_res if thresholds == thresh and d < 10]
            av = np.average(time)
            ci = 1.960 * np.std(time) / np.sqrt(len(time))
            print(f"{tx_range}, {thresholds}: {av} \N{PLUS-MINUS SIGN} {ci}")
            averages.append(av)
            cis.append(ci)
            overall_times.append(time)

        # fig_ax.boxplot(overall_times)
        fig_ax.boxplot(overall_times, showfliers=False)


def process_thresh_per_range(args):
    f = open("angle_thresh_timing/res_thresh_timing_20220503-074224.csv", "r")
    f.readline()  # First line is the header, ignore it
    res = []

    line = f.readline()
    while line != "":
        line = line.strip()
        # drone_speed,measurement_time,finish_dist,tx_range,dist_thresholds,repl_no,time_taken,num_steps,final_dist
        _, _, _, tx_range, thresholds, _, time, _, final_dist = line.split(",")
        thresh_list = [float(t) for t in thresholds.split(";")]
        res.append((float(tx_range), thresh_list, float(time), float(final_dist)))
        line = f.readline()

    f.close()

    unique_tx_ranges = sorted(list(set([tx_range for tx_range, _, _, _ in res])))
    unique_thresholds = list(thresh for thresh, _ in itertools.groupby(sorted([thresh_list for _, thresh_list, _, _ in res])))

    averages = []
    cis = []

    for range_idx in range(len(unique_tx_ranges)):
        tx_range = unique_tx_ranges[range_idx]
        fig, fig_ax = plt.subplots()
        filtered_res = [(thresh, time, final_dist) for tx_range_, thresh, time, final_dist in res if tx_range_ == tx_range]
        bar_refs = []
        for i in range(len(unique_thresholds)):
            thresholds = unique_thresholds[i]
            time = [t for thresh, t, d in filtered_res if thresholds == thresh and d < 10]
            av = np.average(time)
            ci = 1.960 * np.std(time) / np.sqrt(len(time))
            print(f"{tx_range}, {thresholds}: {av} \N{PLUS-MINUS SIGN} {ci}")
            averages.append(av)
            cis.append(ci)

            bar_pos = i
            bar = fig_ax.bar(bar_pos, av, 0.75, yerr=ci, color=COLOURS[i])
            bar_refs.append(bar)

        fig_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        labels = [string.ascii_uppercase[i] for i in range(len(unique_thresholds))]

        if args.show:
            plt.ylabel("Time (s)")
            plt.title(f"$R={tx_range}$")
            if range_idx == 0:
                lgd = fig.legend(bar_refs, labels, loc="upper center", ncol=3)

        if args.pgf:
            plt.ylabel("Time (s)")
            if range_idx == 0:
                lgd = fig.legend(bar_refs, labels, loc="upper center", ncol=3)
                plt.savefig(f"time_per_steps_range_{tx_range:.0f}.pgf", bbox_extra_artists=(lgd,))
            else:
                plt.savefig(f"time_per_steps_range_{tx_range:.0f}.pgf")
        if args.save:
            plt.ylabel("Time (s)")
            if range_idx == 0:
                lgd = fig.legend(bar_refs, labels, loc="upper center", ncol=3)
                plt.savefig(f"time_per_steps_range_{tx_range:.0f}.png", bbox_extra_artists=(lgd,))
            else:
                plt.savefig(f"time_per_steps_range_{tx_range:.0f}.png")


def process_thresh(args):
    f = open("angle_thresh_timing/res_thresh_timing_20220503-074224.csv", "r")
    f.readline()  # First line is the header, ignore it
    res = []

    line = f.readline()
    while line != "":
        line = line.strip()
        # drone_speed,measurement_time,finish_dist,tx_range,dist_thresholds,repl_no,time_taken,num_steps,final_dist
        _, _, _, tx_range, thresholds, _, time, _, final_dist = line.split(",")
        thresh_list = [float(t) for t in thresholds.split(";")]
        res.append((float(tx_range), thresh_list, float(time), float(final_dist)))
        line = f.readline()

    f.close()

    unique_tx_ranges = sorted(list(set([tx_range for tx_range, _, _, _ in res])))
    unique_thresholds = list(thresh for thresh, _ in itertools.groupby(sorted([thresh_list for _, thresh_list, _, _ in res])))

    averages = []
    cis = []

    fig, fig_ax = plt.subplots()

    bar_pos_base = 0
    bar_refs = []
    for tx_range in unique_tx_ranges:
        filtered_res = [(thresh, time, final_dist) for tx_range_, thresh, time, final_dist in res if tx_range_ == tx_range]
        for i in range(len(unique_thresholds)):
            thresholds = unique_thresholds[i]
            time = [t for thresh, t, d in filtered_res if thresholds == thresh and d < 10]
            av = np.average(time)
            ci = 1.960 * np.std(time) / np.sqrt(len(time))
            print(f"{tx_range}, {thresholds}: {av} \N{PLUS-MINUS SIGN} {ci}")
            averages.append(av)
            cis.append(ci)

            bar_pos = (bar_pos_base * len(unique_thresholds)) + i
            bar = fig_ax.bar(bar_pos, av, 0.75, yerr=ci, color=COLOURS[i])
            bar_refs.append(bar)

            # label = ""
            # for i in range(len(thresholds)):
            #     if i % 3 == 2:
            #         label = label + f"{thresholds[i]:.0f},\n"
            #     else:
            #         label = label + f"{thresholds[i]:.0f},"
            # fig_ax.bar_label(bar, label_type='center', rotation='vertical', labels=[label.strip()[:-1]])

        bar_pos_base += 1

    fig_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    fig_ax.set_xticks([(len(unique_thresholds) * i) + ((len(unique_thresholds) - 1) / 2) for i in range(len(unique_tx_ranges))])
    fig_ax.set_xticklabels(unique_tx_ranges)

    labels = [string.ascii_uppercase[i] for i in range(len(unique_thresholds))]
    lgd = fig.legend(bar_refs[:len(unique_thresholds)], labels, loc="upper center", ncol=3)





def parse_args():
    parser = argparse.ArgumentParser(usage="%(prog)s...",
                                     description="Produce plots of the angle threshold data",
                                     add_help=True)
    parser.add_argument("-s", "--show", help="Show the plots before closing", action="store_true")
    parser.add_argument("-S", "--save", help="Save the plots before closing", action="store_true")
    parser.add_argument("-p", "--pgf", help="Save the plots before closing in the PGF format", action="store_true")
    parser.add_argument("-P", "--per-range", help="Plot each range on a different figure", action="store_true")
    parser.add_argument("-B", "--box-plot", help="Use a box plot rather than bar graph", action="store_true")
    parser.add_argument("-e", "--ecdf", help="Use the ECDF rather than bar graph", action="store_true")
    parser.add_argument("-i", "--inset", help="Include zoom-in insets", action="store_true")
    parser.add_argument("-m", "--mark-insets", help="Include markers on zoom-in insets", action="store_true")

    args = parser.parse_args()

    if args.show and args.pgf:
        raise ValueError("Cannot use PGF and show options at the same time")

    return args


def main():
    args = parse_args()
    if args.pgf:
        set_up_latex()

    if args.per_range:
        if args.box_plot:
            process_thresh_per_range_box_plot(args)
        elif args.ecdf:
            process_thresh_per_range_ecdf(args)
        else:
            process_thresh_per_range(args)
    else:
        process_thresh(args)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
