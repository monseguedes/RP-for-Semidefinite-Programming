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
import random_projections as rp
import pickle
from process_DIMACS_data import Graph_File
import os
from generate_graphs import Graph
import generate_graphs
import mosek


def stable_set_problem_sdp(graph: Graph, verbose=False):
    """
    Write the polynomial optimization problem for the stable set problem.

    minimize    a
    subject to  sum x_v - a = SOS + sum POLY_j (x_v * x_u) + sum POLY_k (x_v^2 - x_v)

    which can be written as

    minimize    a
    subject to  A_i · X  = c_i
                A_0 · X  = c_0

    where powers of monomials are replaced in A_i and contraints for
    edges are removed. 

    Parameters
    ----------
    graph : Graph
        Graph object.
    verbose : bool
        If True, print the log of the Mosek solver.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    distinct_monomials = graph.distinct_monomials_L1
    # print("Number of distinct monomials: ", len(distinct_monomials)
    # )
    # print("Distinct monomials degree 0: {}".format(
    #     [m for m in distinct_monomials if sum(m) == 0]
    # ))
    # print("Distinct monomials degree 1: {}".format(
    #     [m for m in distinct_monomials if sum(m) == 1]
    # ))
    # print("Length of distinct monomials degree 1: {}".format(
    #     len([m for m in distinct_monomials if sum(m) == 1])
    # ))
    # print("Distinct monomials degree 2: {}".format(
    #     [m for m in distinct_monomials if sum(m) == 2]
    # ))
    # print("Length of distinct monomials degree 2: {}".format(
    #     len([m for m in distinct_monomials if sum(m) == 2])
    # ))
    edges = graph.edges
    A = graph.A
    # print("Length of A: ", len(A.keys()))
    # for monomial in distinct_monomials:
    #     print("A_{}: {}".format(monomial, A[monomial]))

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    time_start = time.time()
    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X = c_i
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            SOS_dot_X = mf.Expr.dot(A[monomial], X)
            M.constraint(
                SOS_dot_X,
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

        # PRINTING RANKS AND NORMS----------------------------------------------
        # ----------------------------------------------------------------------
        # for i, monomial in enumerate(A.keys()):
        #     print(
        #         "Frobenious norm of A{}: {}".format(
        #             i, np.linalg.norm(A[monomial], ord="fro")
        #         )
        #     )
        #     print("Rank of A{}: {}".format(i, np.linalg.matrix_rank(A[monomial])))
        # print("Number of distinct monomials: ", len(distinct_monomials))
        # print(
        #     "Rank of solution matrix: ",
        #     np.linalg.matrix_rank(X_sol.reshape(size_psd_variable, size_psd_variable)),
        # )
        # print(
        #     "Nuclear norm of solution matrix: ",
        #     np.linalg.norm(
        #         X_sol.reshape(size_psd_variable, size_psd_variable), ord="nuc"
        #     ),
        # )
        # print("Frobenious norm of X: ", np.linalg.norm(X_sol.reshape(size_psd_variable, size_psd_variable), ord="fro"))
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        solution = {
            # "X": X_sol,
            # "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
        }

        return solution


def projected_stable_set_problem_sdp(graph, random_projector, verbose=False):
    """
    Write the projected problem for the stable set problem.

    maximise    a - UB sum lbv[i] + LB sum ubv[i]
    subject to  PA_iP · X + lbv[i] - ubv[i] = c_i
                PA_0P · X + b = c_0

    where E is the set of edges and V is the set of vertices.

    Parameters
    ----------
    graph : matrix or list
        Adjacency matrix of the graph or list of edges.
    random_projector : RandomProjector
        Random projector.
    verbose : bool
        If True, print the log of the Mosek solver.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    distinct_monomials = graph.distinct_monomials_L1
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Projecting the data matrices
    A = {}
    for monomial in distinct_monomials:
        A[monomial] = random_projector.apply_rp_map(graph.A[monomial])

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Lower and upper bounds
        lb_variables = M.variable(len(distinct_monomials) - 1, mf.Domain.greaterThan(0))
        ub_variables = M.variable(len(distinct_monomials) - 1, mf.Domain.greaterThan(0))

        # Lower and upper bounds of the dual variables
        epsilon = 0.00000001
        dual_lower_bound = 0 - epsilon
        dual_upper_bound = 1 + epsilon

        ones_vector = np.ones(len(distinct_monomials) - 1)
        ones_vector[0] = 0

        # Objective
        b = M.variable()
        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.add(
                b,
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
        # M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + lbv[i] - ubv[i] = c_i
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            matrix_inner_product = mf.Expr.dot(A[monomial], X)
            difference_slacks = mf.Expr.sub(
                lb_variables.index(i),
                ub_variables.index(i),
            )
            # difference_slacks = 0
            M.constraint(
                mf.Expr.add(matrix_inner_product, difference_slacks),
                mf.Domain.equalsTo(C[monomial]),
            )
           
        # Constraint:
        # A_0 · X + b = c_0
        matrix_inner_product = mf.Expr.dot(A[tuple_of_constant], X)
        M.constraint(
            mf.Expr.add(matrix_inner_product, b),
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
            # "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
        }

        return solution

def stable_set_problem_sdp_extension(graph: Graph, verbose=False):
    """
    Parameters
    ----------
    graph : Graph
        Graph object.
    verbose : bool
        If True, print the log of the Mosek solver.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    distinct_monomials = graph.distinct_monomials_L1
    edges = graph.edges
    A = graph.A
   # Expand all matrices with a row and column of 0s
    for matrix in A.keys():
        A[matrix] = np.vstack((A[matrix], np.zeros(A[matrix].shape[1])))
        A[matrix] = np.column_stack((A[matrix], np.zeros(A[matrix].shape[0])))
    # Add a matrix per entry in the new row and column
    new_length = list(A.values())[0].shape[0]
    for i in range(new_length - 1):
        matrix = np.zeros((new_length, new_length))
        matrix[i, -1] = 1
        A[i] = matrix
    for i in range(new_length - 1):
        matrix = np.zeros((new_length, new_length))
        matrix[-1, i] = 1
        A[i + new_length - 1] = matrix

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = new_length
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Objective: maximize a (scalar)
        A_objective = np.zeros((size_psd_variable, size_psd_variable))
        A_objective[-1, -1] = 1
        M.objective(mf.ObjectiveSense.Maximize, mf.Expr.dot(A_objective, X))

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        # Constraints:
        # Monomial constraints
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            SOS_dot_X = mf.Expr.dot(A[monomial], X)
            M.constraint(
                SOS_dot_X,
                mf.Domain.equalsTo(C[monomial]),
            )
        # Scalar constraint
        A_constant = A[tuple_of_constant]
        A_constant[-1, -1] = 1
        M.constraint(
            mf.Expr.dot(A_constant, X),
            mf.Domain.equalsTo(C[tuple_of_constant]),
        )
        # Last row and column constraints
        for i in range(2 * new_length - 2):
            matrix_inner_product = mf.Expr.dot(A[i], X)
            M.constraint(
                matrix_inner_product,
                mf.Domain.equalsTo(0),
            )
        # for i in range(new_length - 1):
        #     M.constraint(X.index(i, -1), mf.Domain.equalsTo(0))
        #     M.constraint(X.index(-1, i), mf.Domain.equalsTo(0))
                
        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        X_sol = X_sol.reshape(size_psd_variable, size_psd_variable)
        objective = X_sol[-1, -1]
        computation_time = end_time - start_time

        solution = {
            # "X": X_sol,
            "objective": objective,
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
        }

        return solution

def random_constraint_aggregation_sdp(graph, projector, verbose=False):
    """
    Parameters
    ----------
    graph : Graph
        Graph object.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    distinct_monomials = graph.distinct_monomials_L1
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}
    
    # Get new set of rhs by randomly combining previous ones
    C_old = C
    C = {}
    for i in range(projector.k):
        for j, monomial in enumerate(distinct_monomials):
            if monomial != tuple([0 for i in range(len(distinct_monomials[0]))]):
                C[i] = projector.projector[i, j] * C_old[monomial]
       
    A = graph.A
    # Expand all matrices with a row and column of 0s
    for matrix in A.keys():
        A[matrix] = np.vstack((A[matrix], np.zeros(A[matrix].shape[1])))
        A[matrix] = np.column_stack((A[matrix], np.zeros(A[matrix].shape[0])))
    for i in range(list(A.values())[0].shape[0]):
        matrix = np.zeros((list(A.values())[0].shape[0], list(A.values())[0].shape[1]))
        matrix[i, -1] = 1
        A[i] = matrix
    for i in range(list(A.values())[0].shape[1]):
        matrix = np.zeros((list(A.values())[0].shape[0], list(A.values())[0].shape[1]))
        matrix[-1, i] = 1
        A[i + list(A.values())[0].shape[0]] = matrix   
    A_old = A
    A = {
        i: sum(
            [
                projector.projector[i, j] * A[monomial]
                for j, monomial in enumerate(distinct_monomials)
            ]
        )
        for i in range(projector.k)
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = list(A.values())[0].shape[0]
        X = M.variable(mf.Domain.inPSDCone(list(A.values())[0].shape[0]))

        # Objective
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        # Constraints:
        # A_i · X  = c_i
        for i in A.keys():
            matrix_inner_product = mf.Expr.dot(A[i], X)
            M.constraint(
                matrix_inner_product,
                mf.Domain.equalsTo(C[i]),
            )
           
        # Constraint:
        # A_0 · X + b = c_0
        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        matrix_inner_product = mf.Expr.dot(A_old[tuple_of_constant], X)
        M.constraint(
            mf.Expr.add(matrix_inner_product, b),
            mf.Domain.equalsTo(C_old[tuple_of_constant]),
        )

        print("Number of constraints: ", len(A.keys()) + 1)

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        # Optimize
        # try:
        start_time = time.time()
        M.solve()
        end_time = time.time()
            # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        objective = M.primalObjValue()
        size_psd_variable = int(np.sqrt(X_sol.shape[0]))
        computation_time = end_time - start_time

        # except:
        #     print("Mosek error")
        #     X_sol = None
        #     b_sol = None
        #     objective = None
        #     size_psd_variable = None
        #     computation_time = None

        solution = {
            # "X": X_sol,
            # "b": b_sol,
            "objective": objective,
            "computation_time": computation_time,
            # "no_linear_variables": no_linear_variables,
            "size_psd_variable": size_psd_variable,
        }

        return solution


def single_graph_results(graph, type="sparse", project="variables", range=(0.4, 0.8), iterations=5):
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
        "\n{: <12} {: >10} {: >8} {: >8}".format(
            "Type", "Size X", "Value", "Time"
        )
    )
    print("-" * 80)

    print(
        "{: <12} {: >10} {: >8.2f} {: >8.2f}".format(
            "Original",
            sdp_solution["size_psd_variable"],
            sdp_solution["objective"],
            sdp_solution["computation_time"],
        )
    )


    matrix_size = graph.graph.shape[0] + 1
    for rate in np.linspace(range[0], range[1], iterations):
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type, seed=seed
        )
        rp_solution = projected_stable_set_problem_sdp(
            graph, random_projector, verbose=False
        )

        print(
            "{: <12.2f} {: >10} {: >8.2f} {: >8.2f}".format(
                rate,
                rp_solution["size_psd_variable"],
                rp_solution["objective"],
                rp_solution["computation_time"],
            )
        )

    print()


def combination_of_graphs_results(graphs_list, rate=0.7, type="sparse"):
    """
    Get the results for a combination of graphs.

    Parameters
    ----------
    graphs_list : list
        List of Graph objects.
    projector : RandomProjector
        Random projector.

    """

    # Solve unprojected stable set problem
    # ----------------------------------------
    print("\n" + "-" * 100)
    print(
        "Results for different graphs for a projector of rate {}".format(
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
    # Open graph from pickle
    # ----------------------------------------
    # directory = "graphs/generalised_petersen_10_2_complement"
    # file_path = directory + "/graph.pkl"
    # with open(file_path, "rb") as file:
    #     graph = pickle.load(file)

    graph = generate_graphs.generate_cordones(5, complement=False, save=False, level=1)
    # graph = generate_graphs.generate_pentagon(complement=True)
    # graph = generate_graphs.generate_generalised_petersen(10, 2, complement=True, save=False, level=1)
    matrix_size = graph.graph.shape[0] + 1
    print("Matrix size: {}".format(matrix_size))

    # single_graph_results(graph, type="sparse")
    results = stable_set_problem_sdp_extension(graph)

    raise SystemExit
    projection = 0.1
    print("No. distinct monomials: ", len(graph.distinct_monomials_L1))
    projector = rp.RandomProjector(round(len(graph.distinct_monomials_L1) * projection), len(graph.distinct_monomials_L1), type="sparse", seed=seed)
    contraintagg = random_constraint_aggregation_sdp(graph, projector, verbose=False)
    print("Objective: ", contraintagg["objective"])


    # graphs_list = []
    # for i in [file for file in os.listdir("graphs") if file != ".DS_Store"]:
    #     file_path = "graphs/" + i + "/graph.pkl"
    #     print("File path: ", file_path)
    #     with open(file_path, "rb") as file:
    #         graph = pickle.load(file)
    #         if graph.n < 20 and graph.filename.split("/")[-1] != "keller4.clq":
    #             graphs_list.append(graph)

    # combination_of_graphs_results(graphs_list)
