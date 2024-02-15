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

    def get_picking_SOS(self):
        """ """

        monomial_matrix = monomials.generate_monomials_matrix(self.n, 2)
        new_matrix = []
        for row in monomial_matrix:
            new_row = []
            for tup in row:
                new_tup = tuple(1 if x in [2,3,4] else x for x in tup)
                new_row.append(new_tup)
            new_matrix.append(new_row)   
        monomial_matrix = new_matrix    

        distinct_monomials = monomials.generate_monomials_up_to_degree(self.n, 2)
        new_vector = []
        for monomial in distinct_monomials:
            new_vector.append(tuple(1 if x in [2,3,4] else x for x in monomial))
        # Remove repeated tuples
        unique_tuples = list(set(new_vector))
        distinct_monomials = unique_tuples

        # Picking monomials from SOS polynomial
        A = {
            monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
            for monomial in distinct_monomials
        }

        self.A = A

    def get_picking_edges(self):
        """ """
        distinct_monomials = monomials.generate_monomials_up_to_degree(self.n, 2)
        # Picking monomials for POLY_(u,v) (x_u * x_v)
        E = {
            monomial: monomials.pick_specific_monomial(
                monomials.edges_to_monomials(self.edges, self.n),
                monomial,
                vector=True,
            )
            for monomial in distinct_monomials
        }

        self.E = E

    def picking_for_level_two(self):
        """
        Get the picking dictionaries for the second level of the polynomial optimization problem.
        """

        monomial_matrix = monomials.second_level_monomial_matrix(self.edges, self.n) 

        self.distinct_monomials_L2 = monomials.second_level_distinct_monomials(self.edges, self.n)

        # Picking monomials of degree 2 or less
        monomials_free_polynomials = [monomial for monomial in self.distinct_monomials_L2 if sum(monomial) <= 2]
        self.monomials_free_polynomials = monomials_free_polynomials
        
        #monomials.generate_monomials_up_to_degree(
        #     self.n, 2
        # )
        # new_vector = []
        # for monomial in monomials_free_polynomials:
        #     new_vector.append(tuple(1 if x in [2,3,4] else x for x in monomial))
        # # Remove repeated tuples
        # unique_tuples = list(set(new_vector))
        # # Remove those that belong to an edge.
        # unique_tuples = [
        #     monomial
        #     for monomial in unique_tuples
        #     if monomial not in self.edges
        # ]
        # monomials_free_polynomials = unique_tuples

        A_L2 = {
            monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
            for monomial in self.distinct_monomials_L2
        }

        self.A_L2 = A_L2

        # Picking monomials for POLY_(u,v) (x_u * x_v)
        E_L2 = {
            monomial: [
                monomials.pick_specific_monomial(
                    tuple(1 if x in [2,3,4] else x for x in 
                    ous.add_tuple_to_tuple_list(
                        monomials.edge_to_monomial(edge, self.n),
                        monomials_free_polynomials,
                    )),
                    monomial,
                    vector=True,
                )
                for edge in self.edges
            ]
            for monomial in self.distinct_monomials_L2
        }
        self.E_L2 = E_L2

        single_monomials = list(monomials.generate_monomials_exact_degree(self.n, 1))
        squared_monomials = [
            monomial
            for monomial in list(monomials.generate_monomials_exact_degree(self.n, 2))
            if any(n == 2 for n in monomial)
        ]

        # Picking monomials for POLY_v (x_v^2)
        V_squared = {
            monomial: [
                monomials.pick_specific_monomial(
                    tuple(1 if x in [2,3,4] else x for x in 
                    ous.add_tuple_to_tuple_list(
                        single_monomials[variable], monomials_free_polynomials
                    )),
                    monomial,
                    vector=True,
                )
                for variable in range(self.n)
            ]
            for monomial in self.distinct_monomials_L2
        }

        # Picking monomials for POLY_v (x_v)
        V = {
            monomial: [
                monomials.pick_specific_monomial(
                    tuple(1 if x in [2,3,4] else x for x in
                    ous.add_tuple_to_tuple_list(
                        single_monomials[variable], monomials_free_polynomials
                    )),
                    monomial,
                    vector=True,
                )
                for variable in range(self.n)
            ]
            for monomial in self.distinct_monomials_L2
        }

        self.V_L2 = V
        self.V_squared_L2 = V_squared

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
    
    raise NotImplementedError("This function is not implemented yet.")

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


def generate_generalised_petersen(n, k):
    """
    Generate a generalised petersen graph.
    """
    petersen = Graph()
    petersen.n = 2*n
    inner_star_edges = [(i, (i+k) % n) for i in range(n)]
    petersen.edges = inner_star_edges

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
    petersen.store_graph("generalised_petersen_{}_{}".format(n, k))

if __name__ == "__main__":
    # generate_pentagon()
    # generate_pentagon_with_legs()
    # generate_triangle()
    # generate_petersen_graph()
    generate_6_wheel_graph()
    # generate_generalised_petersen(6, 2)