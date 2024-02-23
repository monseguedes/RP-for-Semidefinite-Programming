import numpy as np
import networkx as nx
import mosek.fusion as mf
import time
import matplotlib.pyplot as plt
import pickle
from generate_graphs import Graph
import random_projections as rp


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
            constraints.append(M.constraint(X.index(i, i), mf.Domain.equalsTo(1)))

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        X_sol = X_sol.reshape((n, n))

        computation_time = end_time - start_time

        solution = {
            "X_sol": X_sol,
            "computation_time": computation_time,
            "objective": M.primalObjValue(),
            "size_psd_variable": n,
        }

        return solution


def projected_sdp_relaxation(graph, projector, verbose=False, slack=True):
    """ """

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

    projected_A = {i: projector.apply_rp_map(A[i]) for i in range(original_dimension)}

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(n))

        lb_variables = M.variable(original_dimension, mf.Domain.greaterThan(0))
        ub_variables = M.variable(original_dimension, mf.Domain.greaterThan(0))
        ones_vector = np.ones(original_dimension)

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = -100 - epsilon
        dual_upper_bound = 100 + epsilon

        # Objective:
        # M.objective(mf.ObjectiveSense.Maximize, mf.Expr.mul(1 / 4, mf.Expr.dot(L, X)))
        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.add(
                mf.Expr.mul(1 / 4, mf.Expr.dot(L, X)),
                mf.Expr.sub(
                    mf.Expr.mul(
                        dual_lower_bound,
                        mf.Expr.dot(lb_variables, ones_vector),
                    ),
                    mf.Expr.mul(
                        dual_upper_bound,
                        mf.Expr.dot(ub_variables, ones_vector),
                    ),
                ),
            ),
        )

        # Constraints:
        constraints = []
        for i in range(original_dimension):
            difference_slacks = mf.Expr.sub(
                lb_variables.index(i),
                ub_variables.index(i),
            )
            constraints.append(
                M.constraint(
                    mf.Expr.add(mf.Expr.dot(projected_A[i], X), difference_slacks),
                    mf.Domain.equalsTo(1),
                )
            )

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        computation_time = end_time - start_time

        X_sol = X_sol.reshape((n, n))

        solution = {
            "X_sol": X_sol,
            "computation_time": computation_time,
            "objective": M.primalObjValue(),
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
        Graph object.
    type : str
        Type of random projector.

    """

    # Solve unprojected stable set problem
    # ----------------------------------------
    print("\n" + "-" * 80)
    print("Results for a graph with {} vertices".format(graph.n).center(80))
    print("-" * 80)
    print(
        "\n{: <18} {: >10} {: >8} {: >8} {: >8}".format(
            "Type", "Size X", "Value", "Time", "Cut"
        )
    )
    print("-" * 80)

    sdp_solution = sdp_relaxation(graph)
    _, cut = retrieve_solution(sdp_solution["X_sol"], graph.edges)
    print(
        "{: <18} {: >10} {: >8.2f} {: >8.2f} {: >8}".format(
            "SDP Relaxation",
            sdp_solution["size_psd_variable"],
            sdp_solution["objective"],
            sdp_solution["computation_time"],
            cut,
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
        rp_solution = projected_sdp_relaxation(
            graph, random_projector, verbose=False, slack=slack
        )
        # Lift up solution
        lifted_solution = random_projector.lift_solution(rp_solution["X_sol"])
        _, cut = retrieve_solution(lifted_solution, graph.edges)

        print(
            "{: <18.2f} {: >10} {: >8.2f} {: >8.2f} {: >8}".format(
                rate,
                rp_solution["size_psd_variable"],
                rp_solution["objective"],
                rp_solution["computation_time"],
                cut,
            )
        )

    # Solve identity projector
    # ----------------------------------------
    id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    id_rp_solution = projected_sdp_relaxation(
        graph, id_random_projector, verbose=False, slack=False
    )
    _, cut = retrieve_solution(id_rp_solution["X_sol"], graph.edges)
    print(
        "{: <18} {: >10} {: >8.2f} {: >8.2f} {: >8}".format(
            "Identity",
            id_rp_solution["size_psd_variable"],
            id_rp_solution["objective"],
            id_rp_solution["computation_time"],
            cut,
        )
    )

    print()


if __name__ == "__main__":
    # Create a graph
    directory = "graphs/400_vertices_0.2_probability"
    file_path = directory + "/graph.pkl"
    with open(file_path, "rb") as file:
        graph = pickle.load(file)

    # Solve the MaxCut problem using the SDP relaxation method
    solution = sdp_relaxation(graph)

    # Get the results for a single graph
    single_graph_results(graph, type="sparser", range=(0.1, 0.9), iterations=9)
