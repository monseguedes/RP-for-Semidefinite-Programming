
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import seaborn as sns

def plot_quality():
    directory = 'results/maxcut/plot'
    if not os.path.exists(directory):
        print('No results to plot')

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

            quality = results[projector_type][0.1]["objective"] / results["original"]["objective"] * 100
            dict[int(name.split("_")[1])] = quality

    # Make a plot from dictionary
    plt.plot(*zip(*sorted(dict.items())))
    plt.xlabel('Size of the graph')
    plt.ylabel('Quality (%)')
    plt.title('Approximation Quality of Projection')

    plt.savefig("plots/quality.png")

plot_quality()