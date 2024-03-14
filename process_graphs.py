"""
Script to process the graph files from the Max-Cut problem.

The graph files are in the following format:
    - The first row contains two numbers: the number of vertices and the number of edges.
    - The following rows contain the edges of the graph.

"""

import os
import sys
import numpy as np
import pickle

class File:
    def __init__(self, filename):
        self.name = filename.split("/")[-1].split(".")[0]
        self.file = open(filename, "r")
        self.lines = self.file.readlines()
        self.edges = self.get_edges()
        self.weights = self.get_weights()
        self.graph = self.get_adjacency_matrix()
        print("Shape of the graph: ", self.graph.shape)
        self.n = self.get_no_vertices()
        self.e = self.get_no_edges() 


    def get_no_vertices(self):
        """
        Get the first number from the first row of the file
        """
        return int(self.lines[0].split()[0])
    
    def get_no_edges(self):
        """
        Get the second number from the first row of the file
        """
        return int(self.lines[0].split()[1])

    def get_edges(self):
        """
        Get the edges from the file
        """
        edges = []
        for line in self.lines[1:]:
            if len(line.split()) > 0:
                edges.append((int(line.split()[0]) - 1, int(line.split()[1]) - 1))

        self.edges = edges

        return edges
    
    def get_weights(self):
        """
        Get the weights from the file
        """
        weights_dict = {}
        for line in self.lines[1:]:
            if len(line.split()) > 0:
                i, j, w = int(line.split()[0]), int(line.split()[1]), int(line.split()[2])
                weights_dict[(i-1, j-1)] = w
                weights_dict[(j-1, i-1)] = w

        return weights_dict

    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix from the file
        """
        n = self.get_no_vertices()
        adj_matrix = np.zeros((n, n))
        for line in self.lines[1:]:
            if len(line.split()) > 0:
                i, j = int(line.split()[0]), int(line.split()[1])
                adj_matrix[i-1][j-1] = self.weights[(i-1, j-1)] if self.weights else 1
                adj_matrix[j-1][i-1] = self.weights[(j-1, i-1)] if self.weights else 1

        self.graph = adj_matrix

        return adj_matrix

    def store_graph(self, name):
        """
        Store the graph in a folder inside the 'maxcut' folder.

        The folder will be named after the graph file.

        """

        self.file.close()
        self.file = None

        directory = "graphs/maxcut/" + name

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

if __name__ == "__main__":
    # file_name = "graphs/maxcut/G1.txt"
    # file = File(file_name)
    # file.store_graph("G1")

    # for i in range(1, 68):
    #     file_name = "graphs/maxcut/G" + str(i) + ".txt"
    #     file = File(file_name)
    #     file.store_graph("G" + str(i))

    # for graph in os.listdir("graphs/maxcut/rudy"):
    #     file_name = "graphs/maxcut/rudy/" + graph
    #     file = File(file_name)
    #     file.store_graph(graph)

    # for graph in os.listdir("graphs/maxcut/ising"):
    #     file_name = "graphs/maxcut/ising/" + graph
    #     file = File(file_name)
    #     file.store_graph(graph)

    for graph in os.listdir("graphs/maxcut/out"):
        file_name = "graphs/maxcut/out/" + graph
        print(file_name)
        file = File(file_name)
        file.store_graph(graph.strip(".txt"))