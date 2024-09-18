"""

"""

import sys
import time
import numpy as np
import mosek.fusion as mf
import monomials
import pickle
from generate_graphs import Graph
from process_DIMACS_data import Graph_File
import first_level_stable_set as ssp
import random_projections as rp
import generate_graphs


def second_level_stable_set_problem_sdp(graph, verbose=False):
    """
    Write the second level of polynomial optimization problem for the stable set problem.

    minimize    a
    subject to  sum x_v - a = SOS + sum POLY_j (x_v * x_u) + sum POLY_k (x_v^2 - x_v)

    for SOS of degree 4 and POLY_j, POLY_k of degree 2.

    Parameters
    ----------
    graph : Graph
        Graph object.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    distinct_monomials = graph.distinct_monomials_L2
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Picking SOS monomials
    A = graph.A_L2

    # print("Starting Mosek")
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
        # A_i · X  = c_i
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            print(
                "Adding constraints... {}/{}          ".format(
                    i + 1, len(distinct_monomials) - 1
                ),
                end="\r",
            )
            # print("Building constraint for monomial {} out of {}".format(i, len(distinct_monomials)))
            SOS_dot_X = mf.Expr.dot(A[monomial], X)

            constraint = M.constraint(
                SOS_dot_X,
                mf.Domain.equalsTo(C[monomial]),
            )
            constraints.append(constraint)

        # Constraint:
        # A_0 · X + b = c_0
        c0 = M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
        )
        constraints.append(c0)
        time_end = time.time()
        # print("Time to build Mosek model: {}".format(time_end - time_start))

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        print(f"Solving the problem of size {size_psd_variable}         ", end="\r")
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        computation_time = end_time - start_time

        solution = {
            # "X": X_sol,
            # "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
            # "no_linear_variables": "TBC",
            "edges": len(edges),
            "no_constraints": len(constraints) + 1,
        }

        return solution


def projected_second_level_stable_set_problem_sdp(
    graph, projector, verbose=False, slack=True
):
    """ """

    distinct_monomials = graph.distinct_monomials_L2
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    A = graph.A_L2
    A = {monomial: projector.apply_rp_map(A[monomial]) for monomial in A.keys()}

    # print("Starting Mosek")
    time_start = time.time()
    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Lower and upper bounds
        if slack:
            lb_variables = M.variable(
                len(distinct_monomials) - 1, mf.Domain.greaterThan(0)
            )
            ub_variables = M.variable(
                len(distinct_monomials) - 1, mf.Domain.greaterThan(0)
            )

            # Lower and upper bounds of the dual variables
            epsilon = 0.00001
            dual_lower_bound = 0 - epsilon
            dual_upper_bound = 1 + epsilon

        ones_vector = np.ones(len(distinct_monomials) - 1)
        ones_vector[0] = 0

        # Objective: maximize a (scalar)
        b = M.variable()
        if slack:
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
        else:
            M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + lbv[i] - ubv[i] = c_i
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            print(
                "Adding constraints... {}/{}          ".format(
                    i + 1, len(distinct_monomials) - 1
                ),
                end="\r",
            )
            # print("Building constraint for monomial {} out of {}".format(i, len(distinct_monomials)))
            SOS_dot_X = mf.Expr.dot(A[monomial], X)

            if slack:
                difference_slacks = mf.Expr.sub(
                    lb_variables.index(i),
                    ub_variables.index(i),
                )
            else:
                difference_slacks = 0

            c = M.constraint(
                mf.Expr.add(
                    SOS_dot_X,
                    difference_slacks,
                ),
                mf.Domain.equalsTo(C[monomial]),
            )
            constraints.append(c)

        # Constraint:
        # A_0 · X + b  = c_0
        c0 = M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
        )
        constraints.append(c0)
        time_end = time.time()
        # print("Time to build Mosek model: {}".format(time_end - time_start))

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        print(f"Solving the problem of size {size_psd_variable}         ", end="\r")
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        computation_time = end_time - start_time

        if slack:
            linear_variables = 2 * len(constraints) + 1
        else:
            linear_variables = 1

        solution = {
            # "X": X_sol,
            # "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
            # "no_linear_variables": "TBC",
            "edges": len(edges),
            "no_constraints": len(constraints) + 1,
        }

        return solution


def constraint_aggregation(graph, projector, verbose=False):
    """ """

    distinct_monomials = graph.distinct_monomials_L2
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}
    C_old = C.copy()
    for i in range(projector.k):
        C[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            # if monomial != tuple_of_constant:
            C[i] += projector.projector[i, j] * C_old[monomial]

    # Picking SOS monomials
    A = graph.A_L2
    A_old = A.copy()
    for i in range(projector.k):
        A[i] = np.zeros(
            (A_old[tuple_of_constant].shape[0], A_old[tuple_of_constant].shape[1])
        )
        for j, monomial in enumerate(distinct_monomials):
            # if monomial != tuple_of_constant:
            A[i] += projector.projector[i, j] * A_old[monomial]

    # print("Starting Mosek")
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
        # A_i · X  = c_i
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            print(
                "Adding constraints... {}/{}          ".format(
                    i + 1, len(distinct_monomials) - 1
                ),
                end="\r",
            )
            # print("Building constraint for monomial {} out of {}".format(i, len(distinct_monomials)))
            SOS_dot_X = mf.Expr.dot(A[monomial], X)

            constraint = M.constraint(
                SOS_dot_X,
                mf.Domain.equalsTo(C[monomial]),
            )
            constraints.append(constraint)

        # Constraint:
        # A_0 · X + b = c_0
        c0 = M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
        )
        constraints.append(c0)
        time_end = time.time()
        # print("Time to build Mosek model: {}".format(time_end - time_start))

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        print(f"Solving the problem of size {size_psd_variable}         ", end="\r")
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        b_sol = b.level()
        computation_time = end_time - start_time

        solution = {
            # "X": X_sol,
            # "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
            # "no_linear_variables": "TBC",
            "edges": edges,
            "no_constraints": len(constraints) + 1,
        }

        return solution


def projected_dimension(epsilon, probability, ranks_Ai, rank_solution):
    """ """

    sum_ranks = 8 * rank_solution + sum(ranks_Ai)

    epsilon_part = np.exp(2 * (epsilon**2 / 2 - epsilon**3 / 3))

    d = np.log(sum_ranks / probability) * epsilon_part

    print("Projected dimension has to be at least: {}".format(d))


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

    distinct_monomials = graph.distinct_monomials_L2
    edges = graph.edges
    tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Get new set of rhs by randomly combining previous ones
    C_old = C.copy()
    C = {}
    for i in range(projector.k):
        C[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            # if monomial != tuple_of_constant:
            C[i] += projector.projector[i, j] * C_old[monomial]

    A = graph.A_L2
    A_old = A.copy()
    A = {}
    for i in range(projector.k):
        A[i] = np.zeros(
            (A_old[tuple_of_constant].shape[0], A_old[tuple_of_constant].shape[1])
        )
        for j, monomial in enumerate(distinct_monomials):
            # if monomial != tuple_of_constant:
            A[i] += projector.projector[i, j] * A_old[monomial]

    b = {monomial: 0 for monomial in distinct_monomials}
    b[tuple_of_constant] = 1
    b_old = b.copy()
    b = {}
    for i in range(projector.k):
        b[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            b[i] += projector.projector[i, j] * b_old[monomial]

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = list(A.values())[0].shape[0]
        # print("Size of PSD variable: ", size_psd_variable)
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Objective
        gamma = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, gamma)

        # Constraints:
        # A_i · X  = c_i
        for i in A.keys():
            matrix_inner_product = mf.Expr.dot(A[i], X)
            M.constraint(
                mf.Expr.add(matrix_inner_product, mf.Expr.mul(b[i], gamma)),
                mf.Domain.equalsTo(C[i]),
            )

        # print("Number of constraints: ", len(A.keys()) + 1)

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        # Optimize
        try:
            start_time = time.time()
            M.solve()
            end_time = time.time()
            # Get the solution
            X_sol = X.level()
            X_sol = X_sol.reshape(size_psd_variable, size_psd_variable)
            objective = M.primalObjValue()
            computation_time = end_time - start_time

        except:
            # print("Unbounded relaxation")
            X_sol = np.inf
            objective = np.inf
            computation_time = np.inf

        solution = {
            # "X": X,
            "objective": objective,
            "size_psd_variable": size_psd_variable,
            "no_constraints": projector.k,
            "computation_time": computation_time,
            "edges": len(edges),
        }

        return solution


def combined_projection(
    graph, projector_variables, projector_constraints, verbose=False
):
    distinct_monomials = graph.distinct_monomials_L2
    edges = graph.edges
    tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Get new set of rhs by randomly combining previous ones
    C_old = C.copy()
    C = {}
    for i in range(projector_constraints.k):
        C[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            # if monomial != tuple_of_constant:
            C[i] += projector_constraints.projector[i, j] * C_old[monomial]

    A = graph.A_L2
    A = {monomial: projector_variables.apply_rp_map(A[monomial]) for monomial in A.keys()}
    A_old = A.copy()
    A = {}
    for i in range(projector_constraints.k):
        A[i] = np.zeros(
            (A_old[tuple_of_constant].shape[0], A_old[tuple_of_constant].shape[1])
        )
        for j, monomial in enumerate(distinct_monomials):
            # if monomial != tuple_of_constant:
            A[i] += projector_constraints.projector[i, j] * A_old[monomial]

    b = {monomial: 0 for monomial in distinct_monomials}
    b[tuple_of_constant] = 1
    b_old = b.copy()
    b = {}
    for i in range(projector_constraints.k):
        b[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            b[i] += projector_constraints.projector[i, j] * b_old[monomial]

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = list(A.values())[0].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))
        print("Size of PSD variable: ", size_psd_variable)

        # Objective
        gamma = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, gamma)

        # Constraints:
        # A_i · X  = c_i
        for i in A.keys():
            matrix_inner_product = mf.Expr.dot(A[i], X)
            M.constraint(
                mf.Expr.add(matrix_inner_product, mf.Expr.mul(b[i], gamma)),
                mf.Domain.equalsTo(C[i]),
            )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        # Optimize
        try:
            start_time = time.time()
            M.solve()
            end_time = time.time()
            # Get the solution
            X_sol = X.level()
            X_sol = X_sol.reshape(size_psd_variable, size_psd_variable)
            objective = M.primalObjValue()
            computation_time = end_time - start_time

        except:
            X_sol = np.inf
            objective = np.inf
            computation_time = np.inf

        solution = {
            # "X": X,
            "objective": objective,
            "size_psd_variable": size_psd_variable,
            "no_constraints": projector_constraints.k,
            "computation_time": computation_time,
            "edges": len(edges),
        }

        return solution


def alternating_projection_sdp(graph, solution_matrix, objective):
    """
    Alternating projection method to alternate between the
    SDP solution found in the constraint aggregation and the
    original affine subspace.

    """

    iter = 0
    psd_X = solution_matrix
    tuple_of_constant = tuple([0 for i in list(graph.A.keys())[0]])
    b = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in graph.A.keys()}
    old_b = b.copy()
    b.pop(tuple_of_constant)
    Ai = graph.A.copy()
    Ai.pop(tuple_of_constant)
    # b[tuple_of_constant] = b[tuple_of_constant] - objective
    while iter < 1000:
        # Check if linear constraints are satisfied
        for monomial in Ai.keys():
            if np.all(np.dot(Ai[monomial], psd_X) == b[monomial]):
                print(
                    "Linear constraints satisfied for psd matrix at iteration {}".format(
                        iter
                    )
                )
                return psd_X, old_b[tuple_of_constant] - psd_X[0, 0]

        # Project onto the original affine subspace with orthogonal projection
        A = np.array([Ai[monomial].flatten() for monomial in Ai.keys()])
        inverse_AA = np.linalg.inv(A @ A.T)
        b_AX = np.array(
            [b[monomial] - np.trace(Ai[monomial].T @ psd_X) for monomial in Ai.keys()]
        )
        ATAATbAX = A.T @ inverse_AA @ b_AX
        affine_X = psd_X + ATAATbAX.reshape(psd_X.shape)
        affine_X.reshape(psd_X.shape)

        # Check if the projection is psd
        eigenvalues = np.linalg.eigvals(affine_X)
        if np.all(eigenvalues >= -0.0001):
            print("Projection onto affine subspace is psd at iteration {}".format(iter))
            return affine_X, old_b[tuple_of_constant] - affine_X[0, 0]
        else:
            # Project onto the psd cone
            # Spectral decomposition (eigenvalue decomposition)
            eigenvalues, eigenvectors = np.linalg.eig(affine_X)
            V = eigenvectors
            S = np.diag(eigenvalues)
            V_inv = np.linalg.inv(V)
            S = np.maximum(S, 0)
            psd_X = V @ S @ V_inv
            iter += 1

    return psd_X, old_b[tuple_of_constant] - psd_X[0, 0]


def single_graph_results(graph: Graph, 
                         type_variable,
                         type_constraint, 
                         range=(0.1, 0.6), 
                         iterations=5):
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
    print("\n{: <18} {: >10} {: >8} {: >8}".format("Type", "Size X", "Value", "Time"))
    print("-" * 80)

    first_level = ssp.stable_set_problem_sdp(graph, verbose=False)
    print(
        "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
            "Original L1",
            first_level["size_psd_variable"],
            first_level["objective"],
            first_level["computation_time"],
        )
    )

    sdp_solution = second_level_stable_set_problem_sdp(graph, verbose=False)
    print(
        "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
            "Original L2",
            sdp_solution["size_psd_variable"],
            sdp_solution["objective"],
            sdp_solution["computation_time"],
        )
    )
    print("." * 80)
    
    # Solve variable reduction
    # ----------------------------------------
    matrix_size = graph.A_L2[graph.distinct_monomials_L2[0]].shape[0]
    for rate in np.linspace(range[0], range[1], iterations):
        slack = True
        if rate > 0.5:
            slack = True
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type_variable
        )
        rp_solution = projected_second_level_stable_set_problem_sdp(
            graph, random_projector, verbose=False, slack=slack
        )

        print(
            "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                "Projection " + str(round(rate, 2)),
                rp_solution["size_psd_variable"],
                rp_solution["objective"],
                rp_solution["computation_time"],
            )
        )
    print("." * 80)

    # Solve constraint aggregation
    # ----------------------------------------
    no_constraints = len(graph.A_L2.keys())
    for rate in np.linspace(range[0], range[1], iterations):
        random_projector = rp.RandomProjector(
            round(no_constraints * rate), no_constraints, type=type_constraint
        )
        contraintagg = random_constraint_aggregation_sdp(
            graph, random_projector, verbose=False
        )

        print(
            "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                "Constraint " + str(round(rate, 2)),
                contraintagg["size_psd_variable"],
                contraintagg["objective"],
                contraintagg["computation_time"],
            )
        )
    print("." * 80)

    # Solve combined projection
    # ----------------------------------------
    for rate in np.linspace(range[0], range[1], iterations):
        projector_variables = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type_variable
        )
        projector_constraints = rp.RandomProjector(
            round(no_constraints * rate), no_constraints, type=type_constraint
        )
        combined = combined_projection(
            graph, projector_variables, projector_constraints, verbose=False
        )

        print(
            "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                "Combined " + str(round(rate, 2)),
                combined["size_psd_variable"],
                combined["objective"],
                combined["computation_time"],
            )
        )


    # # Solve projected stable set problem
    # # ----------------------------------------
    # id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    # id_rp_solution = projected_second_level_stable_set_problem_sdp(
    #     graph, id_random_projector, verbose=False, slack=False
    # )
    # print(
    #     "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
    #         "Identity",
    #         id_rp_solution["size_psd_variable"],
    #         id_rp_solution["objective"],
    #         id_rp_solution["computation_time"],
    #     )
    # )

    print()


if __name__ == "__main__":
    directory = "graphs/generalised_petersen_20_2_complement"
    file_path = directory + "/graph.pkl"
    with open(file_path, "rb") as file:
        graph = pickle.load(file)

    # graph = generate_graphs.generate_cordones(100, complement=True, save=False, level=1)
    graph = generate_graphs.generate_generalised_petersen(
        10, 2, complement=False, save=False, level=2
    )
    # graph = generate_graphs.generate_jahangir_graph(5, 3, complement=False, save=False, level=2)
    # matrix_size = graph.graph.shape[0] + 1
    # print("Matrix size: {}".format(matrix_size))

    single_graph_results(graph, "sparse", "0.1_density", range=(0.1, 0.9), iterations=9)
    # print("No. distinct monomials: ", len(graph.distinct_monomials_L2))

    # # Alternating projection
    # seed = 1
    # graph = generate_graphs.generate_cordones(6, complement=False, save=False, level=2)
    # # graph = generate_graphs.generate_pentagon(complement=True)
    # # graph = generate_graphs.generate_generalised_petersen(10, 2, complement=True, save=False, level=1)
    # matrix_size = graph.graph.shape[0] + 1
    # print("Matrix size: {}".format(matrix_size))

    # # single_graph_results(graph, type="sparse")
    # results = second_level_stable_set_problem_sdp(graph)
    # print("Objective: ", results["objective"])

    # projection = 0.8
    # print("No. distinct monomials: ", len(graph.distinct_monomials_L2))
    # projector = rp.RandomProjector(
    #     round(len(graph.distinct_monomials_L2) * projection),
    #     len(graph.distinct_monomials_L2),
    #     type="gaussian",
    #     seed=seed,
    # )
    # contraintagg = random_constraint_aggregation_sdp(graph, projector, verbose=False)
    # print("Objective: ", contraintagg["objective"])

    # alternated_X, bound = alternating_projection_sdp(
    #     graph, contraintagg["X"], contraintagg["objective"]
    # )
    # print("Bound: ", bound)
