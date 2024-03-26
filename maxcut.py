"""
This script contains the implementation of the MaxCut problem
using the SDP relaxation method and its projection. 

The script also contains the implementation of the random projection
map for SDP problems, i.e. m_P(A):A --> PAP^T, where P is a random
projection matrix. We solve the following problem:

    max 1/4 * <L, X>
    s.t. X is PSD
         X_ii = 1 for all i

"""

import numpy as np
import networkx as nx
import mosek.fusion as mf
import time
import matplotlib.pyplot as plt
import pickle
from generate_graphs import Graph
import random_projections as rp
from process_graphs import File
import monomials


def laplacian_matrix(graph):
    """
    Returns the Laplacian matrix of a weighted
    graph.

    Parameters
    ----------
    graph : networkx.Graph
        A weighted graph.

    Returns
    -------
    numpy.ndarray
        The Laplacian matrix of the graph.
    """

    graph = nx.from_numpy_array(graph.graph)

    # Get the adjacency matrix of the graph
    A = nx.adjacency_matrix(graph).todense()

    # Get the degree matrix of the graph
    D = np.diag(np.array(A.sum(axis=1)).flatten())

    # Return the Laplacian matrix
    return D - A


def sdp_relaxation(graph):
    """
    Solves the MaxCut problem using the SDP relaxation
    method.

    This problem is formulated as a semidefinite program
    as follows:

    max 1/4 * <L, X>
    s.t. X is PSD
         X_ii = 1 for all i
    
    where L is the Laplacian matrix of the graph.

    Parameters
    ----------
    graph : networkx.Graph
        A weighted graph.

    Returns
    -------
    numpy.ndarray
        The solution to the MaxCut problem.
    """

    # Get the Laplacian matrix of the graph
    L = laplacian_matrix(graph)

    # Get the number of nodes in the graph
    n = L.shape[0]

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(n))

        # Objective:
        M.objective(mf.ObjectiveSense.Maximize, mf.Expr.mul(1 / 4, mf.Expr.dot(L, X)))

        # Constraints:
        constraints = []
        for i in range(n):
            print("Adding constraints... {}/{}          ".format(i + 1, n), end="\r")
            constraints.append(M.constraint(X.index(i, i), mf.Domain.equalsTo(1)))

        start_time = time.time()
        print(f"Solving the problem of size {n}         " , end="\r")
        M.solve()
        end_time = time.time()


        solution = {
            # "X_sol": X.level().reshape((n, n)),
            "computation_time": end_time - start_time,
            "objective": M.primalObjValue(),
            "size_psd_variable": n,
            "edges": len(graph.edges),
        }

        # print("The nuclear norm the solution matrix is: ", np.linalg.norm(solution["X_sol"], "nuc"))
        # print("The frobenius norm of the solution matrix is: ", np.linalg.norm(solution["X_sol"], "fro"))
        # print("The nuclear norm of the laplacian matrix is: ", np.linalg.norm(L, "nuc"))
        # print("The frobenius norm of the laplacian matrix is: ", np.linalg.norm(L, "fro"))


        return solution


def projected_sdp_relaxation(graph, projector, verbose=False, slack=True):
    """ 
    Solves the MaxCut problem using the SDP relaxation method
    and a random projection map.

    This problem is formulated as a semidefinite program
    as follows:

    TODO: Add the problem formulation here.
    
    """

    L = laplacian_matrix(graph)
    original_dimension = L.shape[0]
    L = projector.apply_rp_map(L)
    n = L.shape[0]

    A_matrix = np.zeros((n, n))
    A = {}
    for i in range(original_dimension):
        A_matrix = np.zeros((original_dimension, original_dimension))
        A_matrix[i, i] = 1
        A[i] = A_matrix

    # projected_A = {}
    # time_start = time.time()
    # for i in range(original_dimension):
    #     print("Projecting A matrix... {}/{}".format(i + 1, original_dimension), end="\r")
    #     projected_A[i] = projector.apply_rp_map(A[i])
    # time_end = time.time()
    # print("Traditional rojection of A matrices took: ", time_end - time_start)

    projected_A = {}
    time_start = time.time()
    for i in range(original_dimension):
        print("Projecting A matrix... {}/{}".format(i + 1, original_dimension), end="\r")
        projected_A[i] = np.outer(projector.projector[:, i], projector.projector[:, i])
    time_end = time.time()
    # print("New projection of A matrices took: ", time_end - time_start)


    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(n))

        # lb_variables = M.variable(original_dimension, mf.Domain.greaterThan(0))
        # ub_variables = M.variable(original_dimension, mf.Domain.greaterThan(0))
        # ones_vector = np.ones(original_dimension)

        # # Lower and upper bounds of the dual variables
        # epsilon = 0.00001
        # dual_lower_bound = -1000000000 - epsilon
        # dual_upper_bound =  1000000000 + epsilon

        # Objective:
        M.objective(mf.ObjectiveSense.Maximize, mf.Expr.mul(1 / 4, mf.Expr.dot(L, X)))
        # M.objective(
        #     mf.ObjectiveSense.Maximize,
        #     mf.Expr.add(
        #         mf.Expr.mul(1 / 4, mf.Expr.dot(L, X)),
        #         mf.Expr.sub(
        #             mf.Expr.mul(
        #                 dual_lower_bound,
        #                 mf.Expr.dot(lb_variables, ones_vector),
        #             ),
        #             mf.Expr.mul(
        #                 dual_upper_bound,
        #                 mf.Expr.dot(ub_variables, ones_vector),
        #             ),
        #         ),
        #     ),
        # )

        # Constraints:
        # constraints = []
        for i in range(original_dimension):
            print("Adding constraints... {}/{}          ".format(i + 1, original_dimension), end="\r")
            # difference_slacks = mf.Expr.sub(
            #     lb_variables.index(i),
            #     ub_variables.index(i),
            # )
            # constraints.append(
            #     M.constraint(
            #         mf.Expr.add(mf.Expr.dot(projector.apply_rp_map(A[i]), X), difference_slacks),
            #         mf.Domain.equalsTo(1),
            #     )
            # )
            # M.constraint(
            #         mf.Expr.add(mf.Expr.dot(projected_A[i], X), difference_slacks),
            #         mf.Domain.equalsTo(1),
            #     )
            M.constraint(
                    mf.Expr.dot(projected_A[i], X),
                    mf.Domain.equalsTo(1),
                )
        

        start_time = time.time()
        # Solve the problem
        print(f"Solving the problem of size {n}         " , end="\r")
        try:
            M.solve()
            # Get the solution
            X_sol = X.level()
            X_sol = X_sol.reshape((n, n))
            objective = M.primalObjValue()
        except: 
            X_sol = 0
            print("The projected problem is infeasible")
            objective = 0

        end_time = time.time()

        computation_time = end_time - start_time

        solution = {
            # "X_sol": X_sol,
            "computation_time": computation_time,
            "objective": objective,
            "size_psd_variable": n,
        }

        return solution


def retrieve_solution(solution_matrix, edges):
    """
    Retrieve a solution from the solution matrix.

    Parameters
    ----------
    solution_matrix : numpy.ndarray
        The solution matrix.

    Returns
    -------
    numpy.ndarray
        The solution vector.
    """

    # Get random vector using solution as covariance matrix
    n = solution_matrix.shape[0]
    random_vector = np.random.multivariate_normal(np.zeros(n), solution_matrix)
    # Apply sign function to random vector
    random_vector = np.sign(random_vector)

    # Count how many edges we have between the two sets
    cut = 0
    for edge in edges:
        if random_vector[edge[0]] != random_vector[edge[1]]:
            cut += 1

    return random_vector, cut


def single_graph_results(graph: Graph, type="sparse", range=(0.1, 0.5), iterations=5):
    """
    Get the results for a single graph.

    Parameters
    ----------
    graph : Graph
        Graph object representing the input graph.
    type : str, optional
        Type of random projector. Defaults to "sparse".
    range : tuple, optional
        Range of rates for the random projector. Defaults to (0.1, 0.5).
    iterations : int, optional
        Number of iterations for the random projector. Defaults to 5.
    """

    # Solve unprojected stable set problem
    # ----------------------------------------
    print("\n" + "-" * 80)
    print("Results for a graph with {} vertices".format(graph.n).center(80))
    print("-" * 80)
    print(
        "\n{: <18} {: >10} {: >8} {: >8} {: >8} {: >8} {:>12}".format(
            "Type", "Size X", "Value", "Quality", "Time", "Cut", "Cut Quality"
        )
    )
    print("-" * 80)

    sdp_solution = sdp_relaxation(graph)
    _, opt_cut = retrieve_solution(sdp_solution["X_sol"], graph.edges)
    print(
        "{: <18} {: >10} {: >8.2f} {: >8} {: >8.2f} {: >8} {:>12}".format(
            "SDP Relaxation",
            sdp_solution["size_psd_variable"],
            sdp_solution["objective"],
            "-",
            sdp_solution["computation_time"],
            opt_cut,
            "-",
        )
    )

    matrix_size = sdp_solution["size_psd_variable"]

    for rate in np.linspace(range[0], range[1], iterations):
        slack = True
        if rate > 0.5:
            slack = True
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type
        )
        # print("Density of projector is:" , np.count_nonzero(random_projector.projector) / (random_projector.projector.size))
        rp_solution = projected_sdp_relaxation(
            graph, random_projector, verbose=False, slack=slack
        )
        quality = rp_solution["objective"] / sdp_solution["objective"] * 100
        # Lift up solution
        lifted_solution = random_projector.lift_solution(rp_solution["X_sol"])
        _, cut = retrieve_solution(lifted_solution, graph.edges)

        print(
            "{: <18} {: >10} {: >8.2f} {: >8} {: >8.2f} {: >8} {:>12}".format(
                "Projection " + str(round(rate, 2)),
                rp_solution["size_psd_variable"],
                rp_solution["objective"],
                str(round(quality, 2)) + "%",
                rp_solution["computation_time"],
                cut,
                str(round(cut / opt_cut * 100, 2)) + "%",
            )
        )

    # # Solve identity projector
    # # ----------------------------------------
    # id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    # id_rp_solution = projected_sdp_relaxation(
    #     graph, id_random_projector, verbose=False, slack=False
    # )
    # quality = id_rp_solution["objective"] / sdp_solution["objective"] * 100
    # _, cut = retrieve_solution(id_rp_solution["X_sol"], graph.edges)
    # print(
    #     "{: <18} {: >10} {: >8.2f} {:>8} {: >8.2f} {: >8} {:>12}".format(
    #         "Identity",
    #         id_rp_solution["size_psd_variable"],
    #         id_rp_solution["objective"],
    #         str(round(quality, 2)) + "%",
    #         id_rp_solution["computation_time"],
    #         cut,
    #         str(round(cut / opt_cut * 100, 2)) + "%",
    #     )
    # )

    # print(sdp_solution["X_sol"])

    print()


def comparison_graphs(graphs_list, percentage):
    """
    Compare the results for a list of graphs.

    Parameters
    ----------
    graphs_list : list
        List of Graph objects.
    """

    # Solve unprojected stable set problem
    # ----------------------------------------
    print("\n" + "-" * 110)
    print("Comparison of different graphs for {}% projection".format(int(percentage * 100)).center(110))
    print("-" * 110)
    print(
        "\n {: <6} {: >10} {: >12} {: >12} {: >18} {: >18} {: >12}".format(
            "Graph", "Size", "SDP value", "SDP cuts", "Projection value", "Projection cuts", "Quality"
        )
    )
    print("-" * 110)
    
    for graph in graphs_list:
        sdp_solution = sdp_relaxation(graph)
        # sdp_solution  = {
        #     "X_sol": "-",
        #     "objective": "-",
        #     "size_psd_variable": graph.n,
        # }
        _, opt_cut = retrieve_solution(sdp_solution["X_sol"], graph.edges)
        # opt_cut = "-"
        matrix_size = sdp_solution["size_psd_variable"]
        random_projector = rp.RandomProjector(
            round(matrix_size * percentage), matrix_size, type="sparse"
        )
        rp_solution = projected_sdp_relaxation(graph, random_projector, verbose=True)
        # Lift up solution
        lifted_solution = random_projector.lift_solution(rp_solution["X_sol"])
        _, cut = retrieve_solution(lifted_solution, graph.edges)
        quality = cut / opt_cut * 100
        # quality = "-"
        print(
            "{: <6} {: >10} {: >12.2f} {: >12} {: >18.2f} {: >18} {: >12}".format(
                graph.name,
                sdp_solution["size_psd_variable"],
                sdp_solution["objective"],
                opt_cut,
                rp_solution["objective"],
                cut,
                str(round(quality, 2)) + "%",
            )
        )


if __name__ == "__main__":
    # Create a graph
    directory = "graphs/300_vertices_0.1_probability"
    directory = "graphs/maxcut/G1"
    file_path = directory + "/graph.pkl"
    with open(file_path, "rb") as file:
        graph = pickle.load(file)

    # # Solve the MaxCut problem using the SDP relaxation method
    # solution = sdp_relaxation(graph)

    # Get the results for a single graph
    single_graph_results(graph, type="sparse", range=(0.1, 0.2), iterations=2)

    # # Get the results for a list of graphs
    # list_of_graphs = []
    # for i in range(1, 10):
    #     file_name = "graphs/maxcut/G" + str(i) + ".txt"
    #     file = File(file_name)
    #     list_of_graphs.append(file)

    # comparison_graphs(list_of_graphs, 0.2)
