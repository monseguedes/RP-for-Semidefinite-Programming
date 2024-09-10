"""

"""

import matplotlib.pyplot as plt
import numpy as np
import math


def plot_psd_matrix_size():
    """
    Plot the number of combinations of n choose k.

    """

    x = np.linspace(0, 100, 100)

    # Function to plot is x + 2 choose x
    y = [math.comb(int(i) + 2, int(i)) for i in x]

    plt.plot(x, y)

    plt.xlabel("Number of vertices")
    plt.ylabel("Size of PSD matrix")
    plt.title("Size of PSD matrix for 2nd Level of Stable Set")
    plt.show()


def plot_constraints_growth():
    """
    Plot the number of constraints for a given number of vertices.

    """

    x = np.linspace(0, 100, 100)

    # Function to plot is x choose 2
    y = [math.comb(int(i) + 4, int(i)) for i in x]

    plt.plot(x, y)

    plt.xlabel("Number of vertices")
    plt.ylabel("Number of constraints")
    plt.title("Number of constraints for a given number of vertices")
    plt.show()


plot_psd_matrix_size()
plot_constraints_growth()
