import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import sys
import seaborn as sns
import yaml

SMALL_SIZE = 12
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


def plot_quality(config):
    directory = "results/maxcut/plot"
    if not os.path.exists(directory):
        print("No results to plot")

    dict = {}
    for name in os.listdir(directory):
        with open(f"results/maxcut/plot/{name}", "rb") as file:
            results = pickle.load(file)
            print(results)

            projector_type = config["densities"][results["original"]["size_psd_variable"]][0]

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
        linewidth=1
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Quality (%)")
    # plt.title('Approximation Quality of Projection')
    # plt.hlines(y=100, xmin=500, xmax=4000, color='b', linestyles='dotted')
    plt.xticks([key for key in dict.keys() if key != 500])
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
        transparent=False,
    )

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

plot_quality(config=config)
