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
    for name in [name for name in os.listdir(directory) if "12000" not in name]:
        with open(f"results/maxcut/plot/{name}", "rb") as file:
            results = pickle.load(file)
            print(results)

            # projector_type = config["densities"][results["original"]["size_psd_variable"]][0]
            projector_type = [key for key in results.keys() if key != "original"][0]

            quality = (
                results[projector_type][0.1]["objective"]
                / results["original"]["objective"]
                * 100
            )
            if int(name.split("_")[1]) not in dict:
                dict[int(name.split("_")[1])] = [quality]
            else:
                dict[int(name.split("_")[1])].append(quality)

    def geo_mean(iterable):
        a = np.array(iterable)
        return a.prod() ** (1.0 / len(a))

    # Take geometric mean of quality
    for key in dict.keys():
        dict[key] = geo_mean(dict[key])

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

def unit_sphere_projection_percentage(name):
    directory = f"results/unit_sphere/{name}.pkl"
    if not os.path.exists(directory):
        print("No results to plot")

    with open(directory, "rb") as file:
        results = pickle.load(file)

    # Pick projector type from config
    matrix_size = results["original"]["size_psd_variable"]
    pick_key = [key for key in config["densities"] if matrix_size in range(int(key.split(",")[0]), int(key.split(",")[1]))]
    type_variable = config["densities"][pick_key[0]][0]
    no_constraints = results["original"]["no_constraints"]
    pick_key = [key for key in config["densities"] if no_constraints in range(int(key.split(",")[0]), int(key.split(",")[1]))]
    type_constraints = config["densities"][pick_key[0]][0]
    projections = list(results[(type_variable, type_constraints)].keys())

    dict_variable = {} 
    dict_constraints = {} 
    dict_combined = {}
    for projection in projections:
        dict_variable[projection] = results[(type_variable, type_constraints)][projection]["variable_reduction"]["objective"]
        dict_constraints[projection] = results[(type_variable, type_constraints)][projection]["constraint_aggregation"]["objective"]
        dict_combined[projection] = results[(type_variable, type_constraints)][projection]["combined_projection"]["objective"]

    print(dict_constraints)
    # Make a plot from dictionary
    plt.plot(
        *zip(*sorted(dict_variable.items())),
        marker="o",
        color="red",
        markersize=3,
        linewidth=1,
    )

    plt.plot(
        *zip(*sorted(dict_constraints.items())),
        marker="o",
        color="blue",
        markersize=3,
        linewidth=1,
    )

    plt.plot(
        *zip(*sorted(dict_combined.items())),
        marker="o",
        color="green",
        markersize=3,
        linewidth=1,
    )

    # Plot horizontal dashed line with original
    plt.hlines(y=results["original"]["objective"], xmin=projections[0], xmax=projections[-1], color='black', linestyles='dotted')

    plt.legend(["Variable reduction", "Constraint aggregation", "Combined projection", "Original SDP"])

    plt.xlabel("Projection (%)")
    plt.ylabel("Bound")
    # plt.title('Approximation Quality of Projection')
    # plt.hlines(y=100, xmin=500, xmax=4000, color='b', linestyles='dotted')
    plt.xticks([key for key in dict_variable.keys()])
    plt.savefig(
        "plots/unit_sphere_percentage.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        "plots/unit_sphere_percentage.png",
        bbox_inches="tight",
        dpi=300,
        transparent=False,
    )


with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# plot_quality(config=config)
unit_sphere_projection_percentage("form-4-10-3")
