"""

"""

import monomials
import os
import pickle
import numpy as np
import optimization_unit_sphere as ous
import matplotlib.pyplot as plt
import networkx as nx


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
            
        complement = []

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (i, j) not in self.edges and (j, i) not in self.edges:
                    complement.append((i, j))

        self.edges = complement
         
    def get_picking_SOS(self, verbose=False):
        """ """

        monomial_matrix = monomials.stable_set_monomial_matrix(self.edges, self.n, level=1)

        distinct_monomials = monomials.stable_set_distinct_monomials(self.edges, self.n, level=1)
        self.distinct_monomials_L1 = distinct_monomials

        if verbose:
            print("Building Ai matrices for level 1")
        # Picking monomials from SOS polynomial
        A = {
            monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
            for monomial in distinct_monomials
        }
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
        monomial_matrix = monomials.stable_set_monomial_matrix(self.edges, self.n, level=2) 
        if verbose:
            print('SIZE OF MONOMIAL MATRIX:', len(monomial_matrix))
            print("Done building monomial matrix for level 2")
            print("Building distinct monomials for level 2")
        self.distinct_monomials_L2 = monomials.stable_set_distinct_monomials(self.edges, self.n, level=2)
        if verbose:
            print("Done building distinct monomials for level 2")

        # # Picking monomials of degree 2 or less
        # monomials_free_polynomials = [monomial for monomial in self.distinct_monomials_L2 if sum(monomial) <= 2]
        # self.monomials_free_polynomials = monomials_free_polynomials

        if verbose:
            print("Building Ai matrices for level 2")
        # Picking SOS monomials
        self.A_L2 = {}
        for i, monomial in enumerate(self.distinct_monomials_L2):
            if verbose:
                print("Picking monomial: {} out of {}".format(i, len(self.distinct_monomials_L2)))
            self.A_L2[monomial] = monomials.pick_specific_monomial(monomial_matrix, monomial)
           
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

def generate_pentagon():
    """
    Generate a pentagon graph.
    """
    pentagon = Graph()
    pentagon.n = 5
    pentagon.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    # Get namtrix representation of the graph
    pentagon.graph = np.array(
        [[0 for i in range(pentagon.n)] for j in range(pentagon.n)]
    )
    for i, j in pentagon.edges:
        pentagon.graph[i][j] = 1
        pentagon.graph[j][i] = 1
    pentagon.get_picking_SOS()
    pentagon.get_picking_edges()
    pentagon.picking_for_level_two()
    pentagon.store_graph("pentagon")


def generate_pentagon_with_legs():
    """
    Generate a pentagon graph with legs.
    """
    pentagon = Graph()
    pentagon.n = 10
    pentagon.edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ]
    # Get namtrix representation of the graph
    pentagon.graph = np.array(
        [[0 for i in range(pentagon.n)] for j in range(pentagon.n)]
    )
    for i, j in pentagon.edges:
        pentagon.graph[i][j] = 1
        pentagon.graph[j][i] = 1
    pentagon.get_picking_SOS()
    pentagon.get_picking_edges()
    pentagon.picking_for_level_two()
    pentagon.store_graph("pentagon_with_legs")


def generate_triangle():
    """
    Generate a triangle graph.
    """
    triangle = Graph()
    triangle.n = 3
    triangle.edges = [(0, 1), (1, 2), (2, 0)]
    # Get namtrix representation of the graph
    triangle.graph = np.array(
        [[0 for i in range(triangle.n)] for j in range(triangle.n)]
    )
    for i, j in triangle.edges:
        triangle.graph[i][j] = 1
        triangle.graph[j][i] = 1
    triangle.get_picking_SOS()
    triangle.get_picking_edges()
    triangle.picking_for_level_two()
    triangle.store_graph("triangle")


def generate_petersen_graph():
    """
    Generate a petersen graph.
    """
    petersen = Graph()
    petersen.n = 10
    petersen.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
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


def generate_cordones(n):
    """
    
    """
    
    Jn = np.ones((n, n))
    In = np.eye(n)
    en = np.ones((n, 1))
    zn = np.zeros((n, 1))
    Zn = np.zeros((n, n), dtype=int)

    A = np.block([[0,  1,  np.transpose(en),   np.transpose(zn),   np.transpose(zn)], 
                  [1,  0,  np.transpose(zn),   np.transpose(en),   np.transpose(zn)],
                  [en, zn, Zn,                 Jn-In,              In],
                  [zn, en, Jn-In,              Zn,                 In], 
                  [zn, zn, In,                 In,                 Zn]])
    
    cordones = Graph()
    cordones.n = A.shape[0]
    cordones.graph = A
    cordones.get_edges_from_matrix()
    cordones.plot_graph()
    cordones.get_picking_SOS(verbose=True)
    cordones.picking_for_level_two(verbose=True)
    cordones.store_graph("cordones_{}".format(n))


def generate_6_wheel_graph():
    """
    
    """
    wheel = Graph()
    wheel.n = 6
    wheel.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4,0), (0, 5), (1,5), (2,5), (3,5), (4,5)]
    # Get matrix representation of the graph
    wheel.graph = np.array(
        [[0 for i in range(wheel.n)] for j in range(wheel.n)]
    )
    for i, j in wheel.edges:
        wheel.graph[i][j] = 1
        wheel.graph[j][i] = 1

    wheel.plot_graph()
    wheel.get_picking_SOS()
    wheel.get_picking_edges()
    wheel.picking_for_level_two()
    wheel.store_graph("6_wheel")


def generate_generalised_petersen(n, k, complement=False):
    """
    Generate a generalised petersen graph.
    """
    petersen = Graph()
    petersen.n = 2*n
    inner_star_edges = [(i, (i+k) % n) for i in range(n)]
    outer_star_edges = [(i, i+1) for i in range(n, 2*n - 1)] + [(2*n - 1, n)]
    connecting_edges = [element for element in zip(range(n + 1),range(n, 2*n ))]
    petersen.edges = inner_star_edges + outer_star_edges + connecting_edges
    if complement:
        petersen.edges_complement_graph()

    # Get matrix representation of the graph
    petersen.graph = np.array(
        [[0 for i in range(petersen.n)] for j in range(petersen.n)]
    )
    for i, j in petersen.edges:
        petersen.graph[i][j] = 1
        petersen.graph[j][i] = 1

    petersen.get_picking_SOS(verbose=True)
    petersen.picking_for_level_two(verbose=True)
    if complement:
        petersen.store_graph("generalised_petersen_{}_{}_complement".format(n, k))
    else:
        petersen.store_graph("generalised_petersen_{}_{}".format(n, k))


def generate_hexagon():
    """
    Generate a hexagon graph.
    """
    hexagon = Graph()
    hexagon.n = 6
    hexagon.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    # Get matrix representation of the graph
    hexagon.graph = np.array(
        [[0 for i in range(hexagon.n)] for j in range(hexagon.n)]
    )
    for i, j in hexagon.edges:
        hexagon.graph[i][j] = 1
        hexagon.graph[j][i] = 1
    hexagon.get_picking_SOS()
    hexagon.get_picking_edges()
    hexagon.picking_for_level_two()
    hexagon.store_graph("hexagon")


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
    prob.get_picking_SOS(verbose=True)
    prob.picking_for_level_two(verbose=True)
    prob.store_graph("{}_vertices_{}_probability".format(n, p))


def generate_connected_imperfect_graph(repetitions, p, type="6_wheel"):
    """
    Generate a graph made of randomly connecting imperfect graph.

    Parameters
    ----------
    type : str
        Type of graph to connect.
    repetitions : int
        Number of repetitions of the graph.
    d : int
        Probability of the connections.

    Returns
    -------
    matrix
        Adjacency matrix of the graph.
    """

    np.random.seed(0)

    if type == "6_wheel":
        initial_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4,0), (0, 5), (1,5), (2,5), (3,5), (4,5)]
        edges = []
        for i in range(1, repetitions):
            new_edges = [(x + 6*i, y + 6*i) for x, y in initial_edges]
            edges += new_edges
        edges = initial_edges + edges   

    # Loop over all vertices and connect them with probability p to the vertices of other graphs
    for i in range(repetitions):
        for vertex in range(i*6, (i+1)*6):
            print("Vertex:", vertex)
            for j in [graph for graph in range(repetitions) if graph not in [i]]:
                print("Graph:", j)
                for vertex2 in range(j*6, (j+1)*6):
                    print("Vertex2:", vertex2)
                    if np.random.rand() < p:
                        edges.append((vertex, vertex2))
    
    graph = Graph()
    graph.n = 6*repetitions 
    graph.edges = edges
    
    # Get matrix representation of the graph
    graph.graph = np.array(
        [[0 for i in range(graph.n)] for j in range(graph.n)]
    )
    for i, j in graph.edges:
        graph.graph[i][j] = 1
        graph.graph[j][i] = 1

    graph.plot_graph()  
    graph.get_picking_SOS()
    graph.picking_for_level_two(verbose=True)
    graph.store_graph("connected_{}_{}_{}".format(type, repetitions, p))


if __name__ == "__main__":
    # generate_pentagon()
    # generate_pentagon_with_legs()
    # generate_triangle()
    # generate_petersen_graph()
    # generate_6_wheel_graph()
    generate_generalised_petersen(30, 2, complement=True)
    # generate_cordones(7)
    # generate_probability_graph(30, 0.6, seed=0)
    # generate_connected_imperfect_graph(5, 0.4)

