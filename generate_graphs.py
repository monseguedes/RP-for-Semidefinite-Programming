"""

"""

import monomials
import os
import pickle
import numpy as np
import optimization_unit_sphere as ous
import matplotlib.pyplot as plt
import networkx as nx
import time
import random


class Graph:
    def __init__(self):
        self.n = None
        self.edges = []
        self.graph = None
        self.A = None
        self.E = None
        self.A_L2 = None
        self.E_L2 = None
        self.V_L2 = None
        self.V_squared__L2 = None

    def edges_complement_graph(self):
        """
        Get the complement of the graph.

        """

        seed = 0
        random.seed(seed)

        complement = []

        print(sorted(self.edges))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (i, j) not in sorted(self.edges) + complement and (j, i) not in sorted(self.edges) + complement:
                    complement.append((i, j))
        self.edges = complement


    def get_picking_SOS(self, verbose=False):
        """ """

        if verbose:
            print("Building monomial matrix for level 1")
        monomial_matrix = monomials.stable_set_monomial_matrix(
            self.edges, self.n, level=1
        )
        if verbose:
            print("SIZE OF MONOMIAL MATRIX:", len(monomial_matrix))
            print("Done building monomial matrix for level 1")
            print("Building distinct monomials for level 1")
        distinct_monomials = monomials.stable_set_distinct_monomials(
            self.edges, self.n, level=1
        )
        self.distinct_monomials_L1 = distinct_monomials
        if verbose:
            print("Done building distinct monomials for level 1")

        if verbose:
            print("Building Ai matrices for level 1")
        # Picking monomials from SOS polynomial
        A = {}
        for i, monomial in enumerate(distinct_monomials):
            if verbose:
                print(
                    "Picking monomial: {} out of {}".format(i, len(distinct_monomials)),
                    end="\r",
                )
            A[monomial] = monomials.pick_specific_monomial(monomial_matrix, monomial)
       
        if verbose:
            print("Done building Ai matrices for level 1")

        self.A = A

    def get_picking_edges(self, verbose=False):
        """ """
        pass
        # distinct_monomials = monomials.generate_monomials_up_to_degree(self.n, 2)
        # # Picking monomials for POLY_(u,v) (x_u * x_v)
        # E = {
        #     monomial: monomials.pick_specific_monomial(
        #         monomials.edges_to_monomials(self.edges, self.n),
        #         monomial,
        #         vector=True,
        #     )
        #     for monomial in distinct_monomials
        # }

        # self.E = E

    def picking_for_level_two(self, verbose=False):
        """
        Get the picking dictionaries for the second level of the polynomial optimization problem.
        """
        if verbose:
            print("Building monomial matrix for level 2")
        start = time.time()
        monomial_matrix = monomials.stable_set_monomial_matrix(
            self.edges, self.n, level=2
        )
        print("Time taken building monomial matrix for level 2:", time.time() - start)
        if verbose:
            print("SIZE OF MONOMIAL MATRIX:", len(monomial_matrix))
            print("Building distinct monomials for level 2...")
        self.distinct_monomials_L2 = monomials.stable_set_distinct_monomials(
            self.edges, self.n, level=2
        )
        if verbose:
            print("Done building distinct monomials for level 2")

        if verbose:
            print("Building Ai matrices for level 2")
        # Picking SOS monomials
        start = time.time()
        self.A_L2 = {}
        for i, monomial in enumerate(self.distinct_monomials_L2):
            if verbose:
                print(
                    "Picking monomial: {} out of {}".format(
                        i, len(self.distinct_monomials_L2)
                    ), end="\r"
                )
            self.A_L2[monomial] = monomials.pick_specific_monomial(
                monomial_matrix, monomial
            )
        print("Time taken building Ai matrices for level 2:", time.time() - start)

        if verbose:
            print("Done building Ai matrices for level 2")

    def store_graph(self, name):
        """
        Store the graph in a folder inside the 'graphs' folder.

        The folder will be named after the graph file.

        """

        directory = "graphs/" + name

        if not os.path.exists(directory):
            os.mkdir(directory)

        # Save class object with pickle
        # File path where you want to save the object
        file_path = directory + "/graph.pkl"

        # Open the file in binary mode for writing
        with open(file_path, "wb") as file:
            # Serialize and save the object to the file
            pickle.dump(self, file)

        print("Graph stored in: ", file_path)

    def plot_graph(self):
        """
        Plot the graph.

        Parameters
        ----------
        graph : matrix
            Adjacency matrix of the graph.

        """

        G = nx.Graph(self.graph)
        nx.draw(G, with_labels=True, font_weight="bold")
        plt.show()

    def get_edges_from_matrix(self):
        """
        Get the unique edges of the graph from the adjacency matrix.

        """

        edges = []
        for i in range(self.n):
            for j in range(i, self.n):
                if self.graph[i][j] == 1:
                    edges.append((i, j))
        self.edges = edges

    def get_matrix_from_edges(self):
        """
        Get the adjacency matrix from the edges of the graph.

        """

        graph = np.array([[0 for i in range(self.n)] for j in range(self.n)])
        for i, j in self.edges:
            graph[i][j] = 1
            graph[j][i] = 1
        self.graph = graph


def generate_pentagon(complement=False):
    """
    Generate a pentagon graph.
    """
    pentagon = Graph()
    pentagon.n = 5
    pentagon.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    # Get namtrix representation of the graph
    pentagon.graph = np.array(
        [[0 for i in range(pentagon.n)] for j in range(pentagon.n)]
    )
    for i, j in pentagon.edges:
        pentagon.graph[i][j] = 1
        pentagon.graph[j][i] = 1
    if complement:
        pentagon.edges_complement_graph()
        pentagon.get_matrix_from_edges()
    pentagon.plot_graph()
    pentagon.get_picking_SOS()
    pentagon.get_picking_edges()
    pentagon.picking_for_level_two()
    pentagon.store_graph("pentagon")

    return pentagon


def generate_petersen_graph():
    """
    Generate a petersen graph.
    """
    petersen = Graph()
    petersen.n = 10
    petersen.edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ]
    # Get matrix representation of the graph
    petersen.graph = np.array(
        [[0 for i in range(petersen.n)] for j in range(petersen.n)]
    )
    for i, j in petersen.edges:
        petersen.graph[i][j] = 1
        petersen.graph[j][i] = 1

    petersen.plot_graph()
    petersen.get_picking_SOS()
    petersen.get_picking_edges()
    petersen.picking_for_level_two()
    petersen.store_graph("petersen")


def generate_cordones(n, complement=False, save=False, level=2):
    """ 
    Generate a cordones graph.

    This graph looks like:

    [0, 1, 1, ..., 1, 0, ..., 0, 0, ..., 0]
    [1, 0, 0, ..., 0, 1, ..., 1  0, ..., 0]
    [1, 0, 0, ..., 0, 0, ..., 1, 1, ..., 0]
    [., ., ., ..., ., ., ..., ., ., ..., .]
    [1, 0, 0, ..., 0, 1, ..., 0, 0, ..., 1]
    [0, 1, 0, ..., 1, 0, ..., 0, 1, ..., 0]
    [., ., ., ..., ., ., ..., ., ., ..., 0]
    [0, 1, 1, ..., 0, 0, ..., 0, 0, ..., 1]
    [0, 0, 1, ..., 0, 1, ..., 0, 0, ..., 0] 
    [., ., ., ..., ., ., ..., ., ., ..., 0]
    [0, 0, 0, ..., 1, 0, ..., 1, 0, ..., 0]
    """

    print("Generating cordones graph with n={}...".format(n))

    Jn = np.ones((n, n))
    In = np.eye(n)
    en = np.ones((n, 1))
    zn = np.zeros((n, 1))
    Zn = np.zeros((n, n), dtype=int)

    A = np.block(
        [
            [0, 1, np.transpose(en), np.transpose(zn), np.transpose(zn)],
            [1, 0, np.transpose(zn), np.transpose(en), np.transpose(zn)],
            [en, zn, Zn, Jn - In, In],
            [zn, en, Jn - In, Zn, In],
            [zn, zn, In, In, Zn],
        ]
    )

    cordones = Graph()
    cordones.n = A.shape[0]
    cordones.graph = A
    cordones.get_edges_from_matrix()
    # Make sure that all tuples in edges are ordered (i, j) with i < j
    cordones.edges = [tuple(sorted(edge)) for edge in cordones.edges]
    if complement:
        cordones.edges_complement_graph()
        cordones.get_matrix_from_edges()
    # cordones.plot_graph()
    print("-" * 50)
    print("Picking for level 1...")
    start = time.time()
    cordones.get_picking_SOS(verbose=True)
    print("Time taken building level 1:", time.time() - start)
    print("-" * 50)
    if level == 2:
        print("Picking for level 2...")
        start = time.time()
        cordones.picking_for_level_two(verbose=True)
        print("Time taken building level 2:", time.time() - start)
    if save:
        if complement:
            cordones.store_graph("cordones_{}_complement".format(n))
        else:
            cordones.store_graph("cordones_{}".format(n))
    print("-" * 50)
    print("Graph cordones graph with n={} generated!".format(n))

    return cordones


def generate_generalised_petersen(n, k, complement=False, save=False, level=2):
    """
    Generate a generalised petersen graph.
    """
    petersen = Graph()
    petersen.n = 2 * n
    inner_star_edges = [(i, (i + k) % n) for i in range(n)]
    outer_star_edges = [(i, i + 1) for i in range(n, 2 * n - 1)] + [(2 * n - 1, n)]
    connecting_edges = [element for element in zip(range(n + 1), range(n, 2 * n))]
    petersen.edges = inner_star_edges + outer_star_edges + connecting_edges
    # Make sure that all tuples in edges are ordered (i, j) with i < j
    petersen.edges = [tuple(sorted(edge)) for edge in petersen.edges]
    if complement:
        petersen.edges_complement_graph()
    # Make sure that all tuples in edges are ordered (i, j) with i < j
    petersen.edges = [tuple(sorted(edge)) for edge in petersen.edges]

    # Get matrix representation of the graph
    petersen.graph = np.array(
        [[0 for i in range(petersen.n)] for j in range(petersen.n)]
    )
    for i, j in petersen.edges:
        petersen.graph[i][j] = 1
        petersen.graph[j][i] = 1

    print("Generating peteresen graph with n={} and k={}...".format(n, k))
    print("Picking for level 1...")
    # petersen.plot_graph()
    petersen.get_picking_SOS(verbose=True)
    if level == 2:
        print("Picking for level 2...")
        start = time.time()
        petersen.picking_for_level_two(verbose=True)
        print("Time taken building level 2:", time.time() - start)
    if save: 
        if complement:
            petersen.store_graph("generalised_petersen_{}_{}_complement".format(n, k))
        else:
            petersen.store_graph("generalised_petersen_{}_{}".format(n, k))

    print("Graph petersen graph with n={} k={} generated!".format(n, k))

    return petersen


def generate_probability_graph(n, p, seed=0):
    """
    Generate a random graph with n vertices and edge probability p.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Probability of an edge between any two vertices.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    matrix
        Adjacency matrix of the graph.
    """

    prob = Graph()

    np.random.seed(seed)
    prob.n = n
    prob.graph = np.random.choice([0, 1], size=(n, n), p=[1 - p, p])
    prob.graph = np.triu(prob.graph, k=1) + np.triu(prob.graph, k=1).T

    prob.get_edges_from_matrix()
    prob.plot_graph()
    # prob.get_picking_SOS(verbose=True)
    # prob.picking_for_level_two(verbose=True)
    prob.store_graph("{}_vertices_{}_probability".format(n, p))



if __name__ == "__main__":
    # generate_pentagon()
    # generate_generalised_petersen(30, 2, complement=True)
    generate_cordones(15, complement=True)