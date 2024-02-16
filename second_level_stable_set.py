"""

"""

import sys
import time
import numpy as np
import mosek.fusion as mf
import monomials
import pickle
from generate_graphs import Graph
import stable_set_problem as ssp
import random_projections as rp


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
    monomials_free_polynomials = graph.monomials_free_polynomials
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Picking SOS monomials
    A = graph.A_L2
    # # Picking monomials for POLY_v (x_v * x_u)
    # E = graph.E_L2
    # # Picking monomials for POLY_v (x_v^2)
    # V_squared = graph.V_squared_L2
    # # Picking monomials for POLY_v (x_v)
    # V = graph.V_L2

    # print("Starting Mosek")
    time_start = time.time()
    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # # Polynomials for (x_v * x_u)
        # e = []
        # for i, edge in enumerate(edges):
        #     e.append(M.variable(len(monomials_free_polynomials), mf.Domain.unbounded()))

        # # Polynomials for (x_v^2 - x_v)
        # v = []
        # for i in range(graph.n):
        #     v.append(M.variable(len(monomials_free_polynomials), mf.Domain.unbounded()))

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_j Eij · POLY_j + sum_v V_squared_iv · POLY_v - sum_v V_iv · POLY_v = c_i
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            SOS_dot_X = mf.Expr.dot(A[monomial], X)

            # POLY_edge = 0
            # for j, edge in enumerate(edges):
            #     POLY_edge = mf.Expr.add(POLY_edge, mf.Expr.dot(E[monomial][j], e[j]))

            # POLY_v_squared = 0
            # POLY_v = 0
            # for i in range(graph.n):
            #     POLY_v_squared = mf.Expr.add(
            #         POLY_v_squared, mf.Expr.dot(V_squared[monomial][i], v[i])
            #     )
            #     POLY_v = mf.Expr.add(POLY_v, mf.Expr.dot(V[monomial][i], v[i]))

            # V_squared_minus_V = mf.Expr.sub(POLY_v_squared, POLY_v)

            constraint = M.constraint(
                # mf.Expr.add(mf.Expr.add(SOS_dot_X, POLY_edge), V_squared_minus_V),
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
    
        # Print the frobenious norm of the data matrices A.
        for i, monomial in enumerate(A.keys()):
            print(
                "Frobenious norm of A{}: {}".format(
                    i, np.linalg.norm(A[monomial], ord="fro")
                )
            )
            print("Rank of A{}: {}".format(i, np.linalg.matrix_rank(A[monomial])))
            print("Nuclear norm of A{}: {}".format(i, np.linalg.norm(A[monomial], ord="nuc")))

        print("Number of distinct monomials: ", len(distinct_monomials))
        # Print rank of solution matrix
        print(
            "Rank of solution matrix: ",
            np.linalg.matrix_rank(X_sol.reshape(size_psd_variable, size_psd_variable)),
        )
        # Print the nuclear norm of the solution matrix
        print(
            "Nuclear norm of solution matrix: ",
            np.linalg.norm(
                X_sol.reshape(size_psd_variable, size_psd_variable), ord="nuc"
            ),
        )
        # Print the frobenious norm of the solution matrix
        print(
            "Frobenious norm of solution matrix: ",
            np.linalg.norm(
                X_sol.reshape(size_psd_variable, size_psd_variable), ord="fro"
            ),
        )
        print("Number of constraints: ", len(constraints) + 1)

        # print("Monomials of constraints: {}".format([m for m in distinct_monomials if m != tuple_of_constant]))
        # print("Dual variables of constraints: {}".format([np.round(c.dual().item(), 2) for c in constraints]))
        # print("Dual variable of constraint c0: {}".format(np.round(c0.dual().item(), 2)))

        # # Make a dictionary matching monomial and dual
        # duals = {m: np.round(c.dual().item(), 2) for m, c in zip([m for m in distinct_monomials if m != tuple_of_constant], constraints)}# if np.round(c.dual().item(), 2) > 0.1}
        # print("Duals: ", duals)

        solution = {
            "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
            "no_linear_variables": 1,
        }

        return solution


def projected_second_level_stable_set_problem_sdp(graph, projector, verbose=False):
    """ """

    distinct_monomials = graph.distinct_monomials_L2
    monomials_free_polynomials = graph.monomials_free_polynomials
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    A = graph.A_L2
    A = {monomial: projector.apply_rp_map(A[monomial]) for monomial in A.keys()}

    # # Picking monomials for POLY_v (x_v * x_u)
    # E = graph.E_L2

    # # Picking monomials for POLY_v (x_v^2)
    # V_squared = graph.V_squared_L2

    # # Picking monomials for POLY_v (x_v)
    # V = graph.V_L2

    # print("Starting Mosek")
    time_start = time.time()
    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # # Polynomials for (x_v * x_u)
        # e = []
        # for i, edge in enumerate(edges):
        #     e.append(M.variable(len(monomials_free_polynomials), mf.Domain.unbounded()))

        # # Polynomials for (x_v^2 - x_v)
        # v = []
        # for i in range(graph.n):
        #     v.append(M.variable(len(monomials_free_polynomials), mf.Domain.unbounded()))

        # Lower and upper bounds
        lb_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))
        ub_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = 0 - epsilon
        dual_upper_bound = 1 + epsilon

        ones_vector = np.ones(len(distinct_monomials))
        ones_vector[0] = 0

        # Objective: maximize a (scalar)
        b = M.variable()
        # M.objective(mf.ObjectiveSense.Maximize, b)
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

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_j Eij · POLY_j + sum_v V_squared_iv · POLY_v - sum_v V_iv · POLY_v + lbv[i] - ubv[i] = c_i
        constraints = []
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            SOS_dot_X = mf.Expr.dot(A[monomial], X)

            # POLY_edge = 0
            # for j, edge in enumerate(edges):
            #     POLY_edge = mf.Expr.add(POLY_edge, mf.Expr.dot(E[monomial][j], e[j]))

            # POLY_v_squared = 0
            # POLY_v = 0
            # for i in range(graph.n):
            #     POLY_v_squared = mf.Expr.add(
            #         POLY_v_squared, mf.Expr.dot(V_squared[monomial][i], v[i])
            #     )
            #     POLY_v = mf.Expr.add(POLY_v, mf.Expr.dot(V[monomial][i], v[i]))

            # V_squared_minus_V = mf.Expr.sub(POLY_v_squared, POLY_v)

            difference_slacks = mf.Expr.sub(
                lb_variables.index(i + 1),
                ub_variables.index(i + 1),
            )
            # difference_slacks = 0

            c = M.constraint(
                mf.Expr.add(
                    # mf.Expr.add(mf.Expr.add(SOS_dot_X, POLY_edge), V_squared_minus_V),
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

        solution = {
            "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
            "no_linear_variables": 2 * len(constraints) + 1,
        }

        # print("Dual variables of constraints: {}".format([np.round(c.dual().item(), 2) for c in constraints]))
        # print("Dual variable of constraint c0: {}".format(np.round(c0.dual().item(), 2)))

        # # Make a dictionary matching monomial and dual
        # duals = {m: np.round(c.dual().item(), 2) for m, c in zip([m for m in distinct_monomials if m != tuple_of_constant], constraints)}# if np.round(c.dual().item(), 2) > 0.1}
        # print("Duals: ", duals)
    
        return solution


def single_graph_results(graph, type="sparse"):
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
    sdp_solution = second_level_stable_set_problem_sdp(graph)
    print("\n" + "-" * 80)
    print("Results for a graph with {} vertices".format(graph.n).center(80))
    print("-" * 80)
    print(
        "\n{: <12} {: >10} {: >18} {: >8} {: >8}".format(
            "Type", "Size X", "Linear variables", "Value", "Time"
        )
    )
    print("-" * 80)

    first_level = ssp.stable_set_problem_sdp(graph, verbose=False)
    print(
        "{: <12} {: >10} {: >18} {: >8.2f} {: >8.2f}".format(
            "Original L1",
            first_level["size_psd_variable"],
            first_level["no_linear_variables"],
            first_level["objective"],
            first_level["computation_time"],
        )
    )

    print(
        "{: <12} {: >10} {: >18} {: >8.2f} {: >8.2f}".format(
            "Original L2",
            sdp_solution["size_psd_variable"],
            sdp_solution["no_linear_variables"],
            sdp_solution["objective"],
            sdp_solution["computation_time"],
        )
    )

    matrix_size = monomials.number_of_monomials_up_to_degree(graph.n, 2)

    # Solve projected stable set problem
    # ----------------------------------------
    id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    id_rp_solution = projected_second_level_stable_set_problem_sdp(
        graph, id_random_projector, verbose=False
    )
    print(
        "{: <12} {: >10} {: >18} {: >8.2f} {: >8.2f}".format(
            "Identity",
            id_rp_solution["size_psd_variable"],
            id_rp_solution["no_linear_variables"],
            id_rp_solution["objective"],
            id_rp_solution["computation_time"],
        )
    )

    for rate in np.linspace(0.3, 1, 10):
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type
        )
        rp_solution = projected_second_level_stable_set_problem_sdp(
            graph, random_projector, verbose=False
        )

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


def projected_dimension(epsilon, probability, ranks_Ai, rank_solution):
    """
    """

    sum_ranks = 8 * rank_solution + sum(ranks_Ai)

    epsilon_part = np.exp(2 * (epsilon ** 2 / 2 - epsilon ** 3 / 3))

    d = np.log(sum_ranks / probability) * epsilon_part

    print("Projected dimension has to be at least: {}".format(d))


if __name__ == "__main__":
    directory = "graphs/generalised_petersen_9_2"
    file_path = directory + "/graph.pkl"
    with open(file_path, "rb") as file:
        graph = pickle.load(file)

    first_level = ssp.stable_set_problem_sdp(graph, verbose=False)
    print("First level objective: ")
    print(first_level["objective"])
    solution = second_level_stable_set_problem_sdp(graph, verbose=False)
    print("Second level objective: ")
    print(solution["objective"])

    projected_dimension(0.05, 0.8, [6 for i in range(500)], 66)

    single_graph_results(graph, type="sparse")
