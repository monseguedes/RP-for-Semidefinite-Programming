import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import sys
import seaborn as sns

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
# sns.set_style("whitegrid")


def plot_quality():
    directory = "results/maxcut/plot"
    if not os.path.exists(directory):
        print("No results to plot")

    dict = {}
    for name in os.listdir(directory):
        with open(f"results/maxcut/plot/{name}", "rb") as file:
            results = pickle.load(file)
            print(results)

            if 0 <= results["original"]["size_psd_variable"] <= 1000:
                projector_type = "sparse"
            elif 2000 <= results["original"]["size_psd_variable"] <= 3000:
                projector_type = "0.2_density"
            elif results["original"]["size_psd_variable"] == 4000:
                projector_type = "0.1_density"
            elif 5000 <= results["original"]["size_psd_variable"] <= 6000:
                projector_type = "0.05_density"
            elif 7000 <= results["original"]["size_psd_variable"]:
                projector_type = "0.04_density"

            quality = (
                results[projector_type][0.1]["objective"]
                / results["original"]["objective"]
                * 100
            )
            dict[int(name.split("_")[1])] = quality

    # Make a plot from dictionary
    plt.plot(
        *zip(*sorted(dict.items())),
        marker="o",
        color="black",
        markersize=3,
        linewidth=1,
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Quality (%)")
    # plt.title('Approximation Quality of Projection')
    # plt.hlines(y=100, xmin=500, xmax=4000, color='b', linestyles='dotted')
    plt.savefig(
        "plots/quality.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        "plots/quality.png",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )


plot_quality()
