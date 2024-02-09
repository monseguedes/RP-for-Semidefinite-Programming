"""
Script to solve general SDP problems and their projection
with mosek , i.e.problem of the form

    min <C, X>
    s.t. <A_i, X> = b_i, i = 1, ..., m
         X \succeq 0

We also solve its dual problem, i.e. the problem of the form
    
        max b^T y
        s.t. C - \sum_{i=1}^{m} y_i A_i \succeq 0

The script also contains the implementation of the random projection
map for SDP problems, i.e. m_P(A):A --> PAP^T, where P is a random
projection matrix. We solve the following problem:

    min <m_P(C), Y> + UB * sum(ubv) - LB * sum(lbv)
    s.t. <m_P(A_i), Y> + ubv[i] - lbv[i] = b_i, i = 1, ..., m
         Y \succeq 0

"""

import numpy as np
import mosek.fusion as mf
import time
import sys
import random_projections as rp
import random


def standard_primal(C, A: list, b: list, verbose=False):
    """
    Solves the SDP problem of the form

            min <C, X>
            s.t. <A_i, X> = b_i, i = 1, ..., m
                X \succeq 0

    Parameters
    ----------
    C : numpy.ndarray
        The cost matrix of the objective function.
    A : list
        A list of numpy.ndarray matrices.
    b : list
        A list of scalars.
    verbose : bool, optional
        If True, the solver will print the log. The default is False.

    Returns
    -------
    solution : dict
        A dictionary with the solution of the SDP problem. The keys are:
            - "X": The solution of the SDP problem.
            - "objective": The value of the objective function.
            - "computation_time": The time to solve the problem.

    """

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = A[0].shape[0]
        X = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Objective:
        M.objective(mf.ObjectiveSense.Minimize, mf.Expr.dot(C, X))

        # Constraints:
        for i in range(len(A)):
            M.constraint(mf.Expr.dot(A[i], X), mf.Domain.equalsTo(b[i]))

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        X_sol = X.level()
        computation_time = end_time - start_time

        solution = {
            "X": X_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
        }

        return solution


def standard_dual(C, A: list, b: list):
    """
    Solves the dual SDP problem of the form

            max b^T y
            s.t. C - \sum_{i=1}^{m} y_i A_i \succeq 0

    Parameters
    ----------
    C : numpy.ndarray
        The cost matrix of the objective function.
    A : list
        A list of numpy.ndarray matrices.
    b : list
        A list of scalars.

    Returns
    -------
    solution : dict
        A dictionary with the solution of the dual SDP problem. The keys are:
            - "y": The solution of the dual SDP problem.
            - "objective": The value of the objective function.

    """

    with mf.Model("SDP_dual") as M:
        # Dual variable y
        y = M.variable(len(A), mf.Domain.unbounded())

        # Objective:
        M.objective(mf.ObjectiveSense.Maximize, mf.Expr.dot(b, y))

        # Constraints:
        M.constraint(
            mf.Expr.sub(
                C, mf.Expr.add([mf.Expr.mul(y.index(i), A[i]) for i in range(len(A))])
            ),
            mf.Domain.inPSDCone(),
        )

        # Solve the problem
        M.solve()

        # Get the solution
        y_sol = y.level()

        solution = {
            "y": y_sol,
            "objective": M.primalObjValue(),
        }

        return solution


def random_projection_sdp(C, A: list, b: list, projector):
    """
    Solves the SDP problem of the form

            min <m_P(C), Y> + UB * sum(ubv) - LB * sum(lbv)
            s.t. <m_P(A_i), Y> + ubv[i] - lbv[i] = b_i, i = 1, ..., m
                Y \succeq 0

    Parameters
    ----------
    C : numpy.ndarray
        The cost matrix of the objective function.
    A : list
        A list of numpy.ndarray matrices.
    b : list
        A list of scalars.
    projector : numpy.ndarray
        The random projection matrix.

    Returns
    -------
    solution : dict
        A dictionary with the solution of the SDP problem. The keys are:
            - "Y": The solution of the SDP problem.
            - "objective": The value of the objective function.
            - "computation_time": The time to solve the problem.

    """

    with mf.Model("SDP") as M:
        # PSD variable Y
        size_psd_variable = A[0].shape[0]
        Y = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Slack variables
        ubv = M.variable(len(A), mf.Domain.greaterThan(0))
        lbv = M.variable(len(A), mf.Domain.greaterThan(0))

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = -1 - epsilon
        dual_upper_bound = 1 + epsilon

        difference_objective = mf.Expr.sub(
            mf.Expr.mul(dual_upper_bound, mf.Expr.dot(ubv, np.ones(len(A)))),
            mf.Expr.mul(dual_lower_bound, mf.Expr.dot(lbv, np.ones(len(A)))),
        )

        # Objective:
        M.objective(
            mf.ObjectiveSense.Minimize,
            mf.Expr.add(
                mf.Expr.dot(projector.apply_rp_map(C), Y), difference_objective
            ),
        )

        # Constraints:
        for i in range(len(A)):
            difference_constraint = mf.Expr.sub(ubv.index(i), lbv.index(i))
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(projector.apply_rp_map(A[i]), Y), difference_constraint
                ),
                mf.Domain.equalsTo(b[i]),
            )

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        Y_sol = Y.level()

        solution = {
            "Y": Y_sol,
            "objective": M.primalObjValue(),
            "computation_time": end_time - start_time,
        }

        return solution


if __name__ == "__main__":
    # Generate random data
    random.seed(1)
    np.random.seed(1)
    n = 10
    m = 5
    C = np.random.rand(n, n)
    C = C @ C.T
    A = [np.random.rand(n, n) for _ in range(m)]
    A = [Ai @ Ai.T for Ai in A]
    b = np.random.rand(m)


    # Solve the SDP problem
    solution_primal = standard_primal(C, A, b)
    print("Solution of the SDP problem:")
    print(solution_primal["objective"])
    print("The computation time is:")
    print(solution_primal["computation_time"])
    print("\n")

    # # Solve the dual SDP problem
    # solution_dual = standard_dual(C, A, b)
    # print("Solution of the dual SDP problem:")
    # print(solution_dual["objective"])
    # # print("The bounds of the dual variables are:")
    # # print(solution_dual["y"])
    # print("\n")

    # Solve the SDP problem with random projection
    matrix_size = A[0].shape[0]
    rate = 0.5
    projector = rp.RandomProjector(round(rate * matrix_size), matrix_size, "sparse")
    solution_rp = random_projection_sdp(C, A, b, projector)
    print("Solution of the SDP problem with random projection:")
    print(solution_rp["objective"])
    print("The computation time is:")
    print(solution_rp["computation_time"])
