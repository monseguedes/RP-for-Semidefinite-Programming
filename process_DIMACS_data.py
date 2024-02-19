"""
File to process DIMACS data into a format that can be used by our program.

We want to:
- Check that the number of edges is suitable for our program (less than 500).
- Have a function to extract edges list form the file.
- Store all necessary data in a folder inside the 'graphs' folder.
"""

import os
import numpy as np
import pickle
import monomials
import optimization_unit_sphere as ous


class Graph_File:
    def __init__(self, filename):
        self.filename = filename
        print("Filename: ", self.filename)
        self.edges = self.extract_edges()
        self.num_edges = len(self.edges)
        print("Num Edges: ", self.num_edges)
        self.n = self.get_num_vertices()
        print("Num Vertices: ", self.n)
        self.graph = self.make_matrix(self.edges, self.n)
        print("-" * 20)

    def extract_edges(self):
        """
        Extract edges from the file and store them in a tuple
        in the edges list.

        The file is expected to be in DIMACS format.

        The edges are stored as tuples in the format:
        (1,2)
        (2,3)

        The file is expected to be in the following format:
        p edge 5 10
        e 1 2
        e 2 3

        The first line is the number of vertices and edges.
        The following lines are the edges.

        """

        edges = []

        with open(self.filename, "r") as file:
            for line in file:
                if line[0] == "e":
                    edge = line.split(" ")
                    edges.append((int(edge[1]) - 1, int(edge[2]) - 1))

        return edges


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
        
        return complement
    

    def make_matrix(self, edges, num_vertices):
        """
        Make the adjacency matrix of the graph.

        """

        graph = np.zeros((num_vertices, num_vertices))

        for edge in edges:
            graph[edge[0], edge[1]] = 1
            graph[edge[1], edge[0]] = 1

        return graph

    
    def get_num_vertices(self):
        """
        Get the number of vertices from the file.

        This is in the line
        p edge 5 10

        """

        with open(self.filename, "r") as file:
            for line in file:
                if "p" == line[0]:
                    line = line.split(" ")
                    if line[2].strip():
                        self.n = int(line[2])
                    else:
                        self.n = int(line[3])
                    return self.n


    def store_graph(self):
        """
        Store the graph in a folder inside the 'graphs' folder.

        The folder will be named after the graph file.

        """

        folder_name = self.filename.split("/")[1]
        folder_name = folder_name.split(".cl")[0]
        directory = "graphs/" + folder_name

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

        # # Picking monomials for POLY_(u,v) (x_u * x_v)
        # self.E_L2 = {
        #     monomial: [
        #         monomials.pick_specific_monomial(
        #             tuple(1 if x in [2,3,4] else x for x in 
        #             ous.add_tuple_to_tuple_list(
        #                 monomials.edge_to_monomial(edge, self.n),
        #                 monomials_free_polynomials,
        #             )),
        #             monomial,
        #             vector=True,
        #         )
        #         for edge in self.edges
        #     ]
        #     for monomial in self.distinct_monomials_L2
        # }

        # single_monomials = list(monomials.generate_monomials_exact_degree(self.n, 1))

        # # Picking monomials for POLY_v (x_v^2)
        # self.V_squared_L2 = {
        #     monomial: [
        #         monomials.pick_specific_monomial(
        #             tuple(1 if x in [2,3,4] else x for x in 
        #             ous.add_tuple_to_tuple_list(
        #                 single_monomials[variable], monomials_free_polynomials
        #             )),
        #             monomial,
        #             vector=True,
        #         )
        #         for variable in range(self.n)
        #     ]
        #     for monomial in self.distinct_monomials_L2
        # }

        # # Picking monomials for POLY_v (x_v)
        # self.V_L2 = {
        #     monomial: [
        #         monomials.pick_specific_monomial(
        #             tuple(1 if x in [2,3,4] else x for x in
        #             ous.add_tuple_to_tuple_list(
        #                 single_monomials[variable], monomials_free_polynomials
        #             )),
        #             monomial,
        #             vector=True,
        #         )
        #         for variable in range(self.n)
        #     ]
        #     for monomial in self.distinct_monomials_L2
        # }

    

if __name__ == "__main__":
    files_folder = "DIMACS_all_ascii"
    for file in os.listdir(files_folder):
        print("File: ", file)
        graph = Graph_File(files_folder + "/" + file)

        valid_graphs = []
        if graph.n > 100 and graph.n < 200:
            print("*" * 50)
            print("Graph is suitable for our program.")
            print("*" * 50)
            graph_description = {
                "filename": graph.filename,
                "num_edges": graph.num_edges,
                "num_vertices": graph.n
            }
            valid_graphs.append(graph_description)
            print(valid_graphs)
            # graph.edges_complement_graph()
            graph.get_picking_SOS(verbose=True)
            # graph.picking_for_level_two(verbose=True)
            graph.store_graph()

    print("Valid graphs: ", valid_graphs)

