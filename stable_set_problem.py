"""
This module contains functions to write the polynomial
optimization problem for the stable set problem.

minimize    sum x_v
subject to  x_u * x_v = 0 for all (u,v) in E
            x_v^2 = x_v for all v in V

where E is the set of edges and V is the set of vertices.

We do this using the SOS method, i.e. we solve the problem

minimize    a
subject to  sum x_v - a = SOS + sum POLY_j (x_v * x_u) + sum POLY_k (x_v^2 - x_v)

"""

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import monomials
import sys
import mosek.fusion as mf
import time


def get_list_of_edges(graph, full=False):
    """
    Get the list of edges of the graph.

    Parameters
    ----------
    graph : matrix
        Adjacency matrix of the graph.

    Returns
    -------
    list
        List of edges of the graph.
    """

    n = graph.shape[0]
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i, j] == 1:
                edges.append((i, j))

    if full:
        edges_dict = {}
        for i in range(n):
            for j in range(i + 1, n):
                edges_dict[(i, j)] = 0
        for edge in edges:
            edges_dict[edge] = 1

        return edges_dict

    return edges


def generate_graph(n, p, seed=0):
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

    # Set the random seed
    np.random.seed(seed)

    # Generate random values for edges
    edges = np.random.rand(n, n)

    # Create a mask for edges based on probability
    graph = (edges < p).astype(int)

    # Make sure the diagonal is zero (no self-loops)
    np.fill_diagonal(graph, 0)

    # Make the matrix symmetric
    graph = np.triu(graph) + np.triu(graph, 1).T

    return graph


class Graph:
    def __init__(self, n, p, seed=0):
        self.n = n
        self.p = p
        self.graph = generate_graph(self.n, self.p, seed)
        self.edges = get_list_of_edges(self.graph)

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


def stable_set_mip(graph):
    raise NotImplementedError


def stable_set_problem_sdp(graph: Graph, verbose=False):
    """
    Write the polynomial optimization problem for the stable set problem.

    minimize    a
    subject to  sum x_v - a = SOS + sum POLY_j (x_v * x_u) + sum POLY_k (x_v^2 - x_v)

    which can be written as

    minimize    a
    subject to  A_i · X + sum_j∈E B_ij · POL_j(x_u * x_v) + sum_v∈V C_iv · POL_v(x_v2) - D_iv · POLv(x_v) = c_i
                A_0 · X + sumj∈E B_0j · POL_j(x_u * x_v) + sum_v∈V C_0v · POL_v(x_v2) - D_0v · POLv(x_v) = c_0

    where E is the set of edges and V is the set of vertices.

    NOTE: WE ARE ONLY SOLVING THE FIRST LEVEL OF THE HIERARCHY, WHICH MEANS THAT
    ALL POL CAN BE AT MOST DEGREE 1.

    Parameters
    ----------
    graph : Graph
        Graph object.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    monomial_matrix = monomials.generate_monomials_matrix(graph.n, 2)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    degree_1_monomials = monomials.generate_monomials_exact_degree(graph.n, 1)
    degree_2_monomials = [
        monomial
        for monomial in monomials.generate_monomials_exact_degree(graph.n, 2)
        if any(n == 2 for n in monomial)
    ]

    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Picking monomials from SOS polynomial
    A = {
        monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_(u,v) (x_u * x_v)
    E = {
        monomial: monomials.pick_specific_monomial(
            monomials.edges_to_monomials(edges, graph.n), monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v^2)
    V_squared = {
        monomial: monomials.pick_specific_monomial(
            degree_2_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v)
    V = {
        monomial: monomials.pick_specific_monomial(
            degree_1_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Constant for (x_v * x_u)
        e = M.variable(len(graph.edges), mf.Domain.unbounded())

        # Constant for (x_v^2 - x_v)
        v = M.variable(graph.n, mf.Domain.unbounded())

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v = c_i
        for monomial in [m for m in distinct_monomials if m != tuple_of_constant]:
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(A[monomial], X),
                    mf.Expr.add(
                        mf.Expr.dot(E[monomial], e),
                        mf.Expr.sub(
                            mf.Expr.dot(V_squared[monomial], v),
                            mf.Expr.dot(V[monomial], v),
                        ),
                    ),
                ),
                mf.Domain.equalsTo(C[monomial]),
            )

        # Constraint:
        # A_0 · X + b = c_0
        M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
        )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        computation_time = end_time - start_time

        solution = {
            "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
        }

        return solution


def projected_stable_set_problem_sdp(graph: Graph, verbose=False):
    """
    Write the projected problem for the stable set problem.

    minimize    a
    subject to  A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v
                + lbv[i] - ubv[i] = c_i
                A_0 · X + b + lbv[0] - ubv[0] = c_0

    where E is the set of edges and V is the set of vertices.

    NOTE: WE ARE ONLY SOLVING THE FIRST LEVEL OF THE HIERARCHY, WHICH MEANS THAT
    ALL POL CAN BE AT MOST DEGREE 1.

    Parameters
    ----------
    graph : matrix or list
        Adjacency matrix of the graph or list of edges.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    monomial_matrix = monomials.generate_monomials_matrix(graph.n, 2)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    degree_1_monomials = monomials.generate_monomials_exact_degree(graph.n, 1)
    degree_2_monomials = [
        monomial
        for monomial in monomials.generate_monomials_exact_degree(graph.n, 2)
        if any(n == 2 for n in monomial)
    ]
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Picking monomials from SOS polynomial
    A = {
        monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_(u,v) (x_u * x_v)
    E = {
        monomial: monomials.pick_specific_monomial(
            monomials.edges_to_monomials(edges, graph.n), monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v^2)
    V_squared = {
        monomial: monomials.pick_specific_monomial(
            degree_2_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v)
    V = {
        monomial: monomials.pick_specific_monomial(
            degree_1_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Constant for (x_v * x_u)
        e = M.variable(len(graph.edges), mf.Domain.unbounded())

        # Constant for (x_v^2 - x_v)
        v = M.variable(graph.n, mf.Domain.unbounded())

        # Lower and upper bounds
        lb_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))
        ub_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = 0 - epsilon
        dual_upper_bound = 1 + epsilon

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.add(
                b,
                mf.Expr.sub(
                    mf.Expr.mul(
                        dual_lower_bound,
                        mf.Expr.dot(lb_variables, np.ones(len(distinct_monomials))),
                    ),
                    mf.Expr.mul(
                        dual_upper_bound,
                        mf.Expr.dot(ub_variables, np.ones(len(distinct_monomials))),
                    ),
                ),
            ),
        )

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v = c_i
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            matrix_inner_product = mf.Expr.dot(A[monomial], X)
            difference_slacks = mf.Expr.sub(
                lb_variables.index(i + 1),
                ub_variables.index(i + 1),
            )
            M.constraint(
                mf.Expr.add(
                    matrix_inner_product,
                    mf.Expr.add(
                        mf.Expr.dot(E[monomial], e),
                        mf.Expr.sub(
                            mf.Expr.dot(V_squared[monomial], v),
                            mf.Expr.dot(V[monomial], v),
                        ),
                    ),
                ),
                mf.Domain.equalsTo(C[monomial]),
            )

        # Constraint:
        # A_0 · X + b = c_0
        M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
        )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        computation_time = end_time - start_time

def random_constraint_aggregation_sdp(graph: Graph, projector, verbose=False):
    """
    TODO: ADD

    Parameters
    ----------
    graph : Graph
        Graph object.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """


    monomial_matrix = monomials.generate_monomials_matrix(graph.n, 2)

    distinct_monomials = monomials.generate_monomials_up_to_degree(graph.n, 2)

    degree_1_monomials = list(monomials.generate_monomials_exact_degree(graph.n, 1))

    degree_2_monomials = [
        monomial
        for monomial in list(monomials.generate_monomials_exact_degree(graph.n, 2))
        if any(n == 2 for n in monomial)
    ]

    edges = graph.edges

    # rate = 0.7
    # projector = rp.RandomProjector(round(rate * len(distinct_monomials)), len(distinct_monomials), type=type)

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}
    # Get new set of rhs by randomly combining previous ones
    C_old = C
    C = {
        i: sum(
            [
                projector.projector[i, j] * C[monomial]
                for j, monomial in enumerate(distinct_monomials)
            ]
        )
        for i in range(projector.k)
    }

    A = graph.A
    A_old = A
    A = {
        i : sum(
            [
                projector.projector[i, j] * A[monomial]
                for j, monomial in enumerate(distinct_monomials)
            ]
        )
        for i in range(projector.k)
    }

    E = graph.E
    E = {
        i : sum(
            [
                projector.projector[i, j] * E[monomial]
                for j, monomial in enumerate(distinct_monomials)
            ]
        )
        for i in range(projector.k)
    }

    # Picking monomials for POLY_v (x_v^2)
    V_squared = {
        monomial: monomials.pick_specific_monomial(
            degree_2_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }
    V_squared = {
        i : sum(
            [
                projector.projector[i, j] * V_squared[monomial]
                for j, monomial in enumerate(distinct_monomials)
            ]
        )
        for i in range(projector.k)
    }

    # Picking monomials for POLY_v (x_v)
    V = {
        monomial: monomials.pick_specific_monomial(
            degree_1_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }
    V = {
        i : sum(
            [
                projector.projector[i, j] * V[monomial]
                for j, monomial in enumerate(distinct_monomials)
            ]
        )
        for i in range(projector.k)
    }

    # print("Starting Mosek")
    time_start = time.time()
    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A_old[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Constant for (x_v * x_u)
        e = M.variable(len(graph.edges), mf.Domain.unbounded())

        # Constant for (x_v^2 - x_v)
        v = M.variable(graph.n, mf.Domain.unbounded())

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v = c_i
        for i in range(projector.k):
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(A[i], X),
                    mf.Expr.add(
                        mf.Expr.dot(E[i], e),
                        mf.Expr.sub(
                            mf.Expr.dot(V_squared[i], v),
                            mf.Expr.dot(V[i], v),
                        ),
                    ),
                ),
                mf.Domain.equalsTo(C[i]),
            )
            # if i % 100 == 0:
            #     print("Constraint {} of {}".format(i, len(distinct_monomials) - 1))

        # Constraint:
        # A_0 · X + b = c_0
        M.constraint(
            mf.Expr.add(mf.Expr.dot(A_old[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C_old[tuple_of_constant]),
        )
        time_end = time.time()
        # print("Time to build Mosek model: {}".format(time_end - time_start))

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        computation_time = end_time - start_time

        no_linear_variables = len(graph.edges) + graph.n + 1
        size_psd_variable = int(np.sqrt(X_sol.shape[0]))

        # print("Number of distinct monomials: ", len(distinct_monomials))
        # # Print rank of solution matrix
        # print(
        #     "Rank of solution matrix: ",
        #     np.linalg.matrix_rank(X_sol.reshape(size_psd_variable, size_psd_variable)),
        # )
        # # Print the nuclear norm of the solution matrix
        # print(
        #     "Nuclear norm of solution matrix: ",
        #     np.linalg.norm(
        #         X_sol.reshape(size_psd_variable, size_psd_variable), ord="nuc"
        #     ),
        # )
        # # Print the frobenious norm of the data matrices A.
        # for i, monomial in enumerate(A.keys()):
        #     print(
        #         "Frobenious norm of A{}: {}".format(
        #             i, np.linalg.norm(A[monomial], ord="fro")
        #         )
        #     )
        #     print("Rank of A{}: {}".format(i, np.linalg.matrix_rank(A[monomial])))

        solution = {
            "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "no_linear_variables": no_linear_variables,
            "size_psd_variable": size_psd_variable,
        }

        return solution


def single_graph_results(graph, type="sparse", project='variables'):
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
    sdp_solution = stable_set_problem_sdp(graph)
    print("\n" + "-" * 80)
    print("Results for a graph with {} vertices".format(graph.n).center(80))
    print("-" * 80)
    print(
        "\n{: <12} {: >10} {: >18} {: >8} {: >8}".format(
            "Type", "Size X", "Linear variables", "Value", "Time"
        )
    )
    print("-" * 80)

    print(
        "{: <12} {: >10} {: >18} {: >8.2f} {: >8.2f}".format(
            "Original",
            sdp_solution["size_psd_variable"],
            sdp_solution["no_linear_variables"],
            sdp_solution["objective"],
            sdp_solution["computation_time"],
        )
    )

    # Solve projected stable set problem
    # ----------------------------------------
    # id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    # id_rp_solution = projected_stable_set_problem_sdp(graph, id_random_projector)
    # print(
    #     "{: <12} {: >10} {: >18} {: >8.2f} {: >8.2f}".format(
    #         "Identity",
    #         id_rp_solution["size_psd_variable"],
    #         id_rp_solution["no_linear_variables"],
    #         id_rp_solution["objective"],
    #         id_rp_solution["computation_time"],
    #     )
    # )

    for rate in np.linspace(0.5, 1, 10):
        if project == 'variables':
            random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type, seed=seed
            )
            rp_solution = projected_stable_set_problem_sdp(
                graph, random_projector, verbose=False
            )
        elif project == 'constraints':
            number_constraints = len(graph.A.keys())
            random_projector = rp.RandomProjector(
            round(number_constraints * rate), number_constraints, type=type, seed=seed
            )
            rp_solution = random_constraint_aggregation_sdp(graph, random_projector, verbose=False)

        increment = rp_solution["objective"] - sdp_solution["objective"]
        # Print in table format with rate as column and then value and increment as other columns

        print(
            "{: <12.2f} {: >10} {: >18} {: >8.2f} {: >8.2f}".format(
                rate,
                rp_solution["size_psd_variable"],
                rp_solution["no_linear_variables"],
                rp_solution["objective"],
                rp_solution["computation_time"],
            )
        )

    print()


def combination_of_graphs_results(graphs_list):
    """
    Get the results for a combination of graphs.

    Parameters
    ----------
    graphs_list : list
        List of Graph objects.
    projector : RandomProjector
        Random projector.

    """

    rate = 0.7

    # Solve unprojected stable set problem
    # ----------------------------------------
    print("\n" + "-" * 100)
    print(
        "Results for different graphs for a projector of dimension {}".format(
            rate
        ).center(100)
    )
    print("-" * 100)
    print(
        "\n{: <20} {: >10} {: >10} {: >10} {: >10} {: >10} {: >10}".format(
            "Graph", "Vertices", "Edges", "SDP Value", "SDP Time", "RP Value", "RP Time"
        )
    )
    print("-" * 100)

    for graph in graphs_list:
        # print(graph.filename.split("/")[-1])
        sdp_solution = stable_set_problem_sdp(graph)
        matrix_size = graph.graph.shape[0] + 1
        projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type="sparse"
        )
        rp_solution = projected_stable_set_problem_sdp(graph, projector)

        print(
            "{: <20} {: >10} {: >10} {: >10.2f} {: >10.2f}  {: >10.2f} {: >10.2f}".format(
                graph.filename.split("/")[-1],
                graph.n,
                graph.num_edges,
                sdp_solution["objective"],
                sdp_solution["computation_time"],
                rp_solution["objective"],
                rp_solution["computation_time"],
            )
        )

    print()


if __name__ == "__main__":
    seed = 2
    # Possible graphs
    # ----------------------------------------
    # graph = Graph(100, 0.5)
    # graph.plot_graph()

    # Open graph from pickle
    # ----------------------------------------
    directory = "graphs/MANN_a9"
    file_path = directory + "/graph.pkl"
    with open(file_path, "rb") as file:
        graph = pickle.load(file)

    matrix_size = graph.graph.shape[0] + 1
    print("Matrix size: {}".format(matrix_size))

    # random_constraint_aggregation_sdp(graph, type="sparse")

    single_graph_results(graph, type="sparse", project='constraints')

    graphs_list = []
    for i in [file for file in os.listdir("graphs") if file != ".DS_Store"]:
        file_path = "graphs/" + i + "/graph.pkl"
        print("File path: ", file_path)
        with open(file_path, "rb") as file:
            graph = pickle.load(file)
            if graph.n < 200 and graph.filename.split("/")[-1] != "keller4.clq":
                graphs_list.append(graph)

    # combination_of_graphs_results(graphs_list)
