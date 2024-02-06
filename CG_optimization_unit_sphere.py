"""
This module contains functions to write the problem
of optimizing a polynomial function over the unit sphere
as a semidefinite program. 

Problem is taken from the paper
- "Optimization Over Structured Subsets of Positive Semidefinite Matrices
via Column Generation" by Amir Ali Ahmadi, Sanjeeb Dash, and Georgina Hall.

"""

import numpy as np
import scipy.special
import mosek.fusion as mf
import monomials
import polynomial_generation
import random_projections
import math
import sys
import time


def get_sphere_polynomial(n, d):
    """
    Generated the polynomial associated with a sphere of dimension n and degree d.
    In other words, the multinomial theorem applied to (\sum_{i=1}^n (x_i^2))^d. We
    store the monomials and their coeefficients.

    Parameters
    ----------
    n : int
        Dimension of the sphere.
    d : int
        Degree of the sphere.

    Returns
    -------
    polynomial : dict
        Dictionary of monomials and their coefficients.

    Examples
    --------
    >>> get_sphere_polynomial(2,2)
    {(0, 0, 0, 2): 1.0, (0, 0, 2, 0): 1.0, (0, 2, 0, 0): 1.0, (2, 0, 0, 0): 1.0, (0, 0, 1, 1): 4.0, (0, 1, 1, 0): 4.0, (1, 1, 0, 0): 4.0, (0, 1, 0, 1): 4.0, (1, 0, 1, 0): 4.0, (1, 0, 0, 1): 4.0, (0, 2, 0, 0): 1.0, (2, 0, 0, 0): 1.0, (0, 0, 0, 2): 1.0}

    """

    # Multinomial expansion of (x_1^2 + ... + x_n^2)^d
    polynomial = {
        tuple: 0
        for tuple in monomials.get_list_of_distinct_monomials(
            monomials.generate_monomials_matrix(n, 2 * d)
        )
    }

    # The coefficients of the multinomial expansion of (x_1 + ... + x_n)^d
    # are given by d!/ (d_1! * ... * d_n!) where d_1 + ... + d_n = d
    # In our case, since all x are squared, we have d_1 + ... + d_n = 2d
    for monomial in polynomial.keys():
        if sum([power for power in monomial]) == 2 * d and all(
            [bool(power % 2 == 0) for power in monomial if power != 0]
        ):
            polynomial[monomial] = scipy.special.factorial(d) / math.prod(
                [scipy.special.factorial(power / 2) for power in monomial]
            )

    return polynomial


def solve_unit_sphere_polynomial_optimization_problem(
    polynomial: polynomial_generation.Polynomial,
    A: dict,
    verbose: bool = False,
):
    """
    Solves a the relaxation of an unconstrained polynomial optimization problem using Mosek.

    max g
    s.t. A_i · X - a * s_i= c_i
         A_0 · X - a * s_0 = c_0
         X is positive semidefinite

    where X is a symmetric matrix, A_i are symmetric matrices that pick coefficients,
    and c_i are the coefficients of f(x), and s_i is the ith coefficient of the sphere.

    Parameters
    ----------
    polynomial : numpy.ndarray
        Polynomial to be optimized.

    Returns
    -------
    solution : numpy.ndarray
        Solution of the polynomial optimization problem.
    bound : float
        Lower bound of the polynomial optimization problem.

    Examples
    --------
    """

    # Get list of all the matrices for picking coefficients.
    monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    sphere_polynomial = get_sphere_polynomial(polynomial.n, polynomial.d / 2)

    # Get the coefficients of the original polynomial
    polynomial = polynomial.polynomial

    # b = list(polynomial.values())
    # print("b:", b)
    # print("Length of b:", len(b))
    # print("Length of A:", len(A))

    with mf.Model("SDP") as M:
        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        m = A[tuple_of_constant].shape[0]

        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(m))

        # Objective: maximize a (scalar)
        a = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, a)

        # Constraint: A_i · X - a * s_i = c_i
        for i in range(len(A)):
            monomial = distinct_monomials[i]
            if verbose:
                print("A[{}]:".format(i))
                monomials.print_readable_matrix(A[monomial])
                print("monomial: {}".format(monomial))
                print("sphere coefficient: {}".format(sphere_polynomial[monomial]))
                print("polynomial coefficient: {}".format(polynomial[monomial]))
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(A[monomial], X),
                    mf.Expr.mul(sphere_polynomial[monomial], a),
                ),
                mf.Domain.equalsTo(polynomial[monomial]),
            )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        # Solve the problem
        M.solve()

        # Get the solution
        X_sol = X.level()
        a_sol = a.level()

    return a_sol, X_sol


def solve_dual_unit_sphere_polynomial_optimization_problem(
    polynomial: polynomial_generation.Polynomial,
    verbose: bool = False,
):
    """
    Solves the dual problem of the polynomial optimization problem over the unit sphere.

    max c^T * y
    s.t. sum y_i * s_i = 1
        - sum A_i y_i is pd

    where A_i are symmetric matrices that pick coefficients,
    and c_i are the coefficients of f(x), and s_i is the ith coefficient of the sphere.

    Parameters
    ----------
    polynomial : numpy.ndarray
        Polynomial to be optimized.

    Returns
    -------
    solution : numpy.ndarray
        Solution of the polynomial optimization problem.
    bound : float
        Lower bound of the polynomial optimization problem.

    """

    # Get list of all the matrices for picking coefficients.
    monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    sphere_polynomial = get_sphere_polynomial(polynomial.n, polynomial.d / 2)
    A = {}
    for monomial in distinct_monomials:
        A[monomial] = monomials.pick_specific_monomial(monomial_matrix, monomial)

    # Get the coefficients of the original polynomial
    polynomial = polynomial.polynomial

    with mf.Model("SDP") as M:
        # Objective: minimize c^T * y
        y = M.variable(len(distinct_monomials))
        M.objective(
            mf.ObjectiveSense.Minimize, mf.Expr.dot(list(polynomial.values()), y)
        )

        # Constraint: sum y_i * s_i = 1
        M.constraint(
            "sphere constraint",
            mf.Expr.dot(list(sphere_polynomial.values()), y),
            mf.Domain.equalsTo(1),
        )

        # Constraint: - sum A_i y_i is pd TODO fix psd to pd.
        no_matrices = len(distinct_monomials)
        M.constraint(
            "psd constraint",
            mf.Expr.sub(
                0,
                mf.Expr.add(
                    [
                        mf.Expr.mul(y.index(i), A[list(polynomial.keys())[i]])
                        for i in range(no_matrices)
                    ]
                ),
            ),
            mf.Domain.inPSDCone(),
        )

        if verbose:
            print(polynomial)
            print(polynomial.values())
            print("sphere_polynomial.values():", list(sphere_polynomial.values()))
            dot_product = mf.Expr.dot(list(sphere_polynomial.values()), y)
            print("dot_product:", dot_product)
            print(sphere_polynomial)
            print(sphere_polynomial.values())
            print(list(sphere_polynomial.values()))
            for i in range(no_matrices):
                print("A[{}]:".format(i))
                monomials.print_readable_matrix(A[list(polynomial.keys())[i]])
                print(sphere_polynomial[list(polynomial.keys())[i]])
                print(polynomial[list(polynomial.keys())[i]])

            # Increase verbosity
            M.setLogHandler(sys.stdout)
            M.setSolverParam("infeasReportAuto", "on")

        # Solve the problem
        M.solve()

        # Get the solution
        y_sol = y.level()
        obj_sol = M.primalObjValue()

    return y_sol, obj_sol


def solve_unprojected_unit_sphere(polynomial: polynomial_generation.Polynomial):
    """
    Solves the problem of optimizing a polynomial over the unit sphere
    without projecting the matrices, as in the CG paper.

    Parameters
    ----------
    polynomial : polynomial_generation.Polynomial
        Polynomial to be optimized.

    Returns
    -------
    solution : numpy.ndarray
        Solution of the sdp relaxation of the optimization problem.
    bound : float
        Lower bound of the polynomial optimization problem.

    """

    monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    A = {}
    for monomial in distinct_monomials:
        A[monomial] = monomials.pick_specific_monomial(monomial_matrix, monomial)

    a_sol, X_sol = solve_unit_sphere_polynomial_optimization_problem(polynomial, A)

    return a_sol, X_sol


def solve_projected_unit_sphere(
    polynomial: polynomial_generation.Polynomial,
    random_projector: random_projections.RandomProjector,
    verbose: bool = False,
):
    """
    Solves the problem of optimizing a polynomial over the unit sphere
    by projecting the matrices.

    Parameters
    ----------
    polynomial : polynomial_generation.Polynomial
        Polynomial to be optimized.

    Returns
    -------
    solution : numpy.ndarray
        Solution of the sdp relaxation of the optimization problem.
    bound : float
        Lower bound of the polynomial optimization problem.

    """

    epsilon = 0.00001
    dual_lower_bound = -1 - epsilon
    dual_upper_bound = 1 + epsilon

    # Get list of all the matrices for picking coefficients.
    monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    sphere_polynomial = get_sphere_polynomial(polynomial.n, polynomial.d / 2)

    # Get the coefficients of the original polynomial
    polynomial = polynomial.polynomial

    A = {}
    for monomial in distinct_monomials:
        A[monomial] = random_projector.apply_rp_map(
            monomials.pick_specific_monomial(monomial_matrix, monomial)
        )

    with mf.Model("SDP") as M:
        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        m = A[tuple_of_constant].shape[0]
        X = M.variable(mf.Domain.inPSDCone(m))

        # Objective: (maximize) a + LB * sum(lbv) - UB * sum(ubv)
        a = M.variable()
        lb_variables = M.variable(
            "lb_variables", len(distinct_monomials), mf.Domain.greaterThan(0)
        )
        ub_variables = M.variable(
            "ub_variables", len(distinct_monomials), mf.Domain.greaterThan(0)
        )

        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.add(
                a,
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

        # Constraint: A_i · X - a * s_i + lbv[i] - ubv[i] = c_i
        for i in range(len(A)):
            monomial = distinct_monomials[i]
            if verbose:
                print("A[{}]:".format(i))
                monomials.print_readable_matrix(A[monomial])
                print("monomial: {}".format(monomial))
                print("sphere coefficient: {}".format(sphere_polynomial[monomial]))
                print("polynomial coefficient: {}".format(polynomial[monomial]))
            M.constraint(
                mf.Expr.add(
                    mf.Expr.add(
                        mf.Expr.dot(A[monomial], X),
                        mf.Expr.mul(sphere_polynomial[monomial], a),
                    ),
                    mf.Expr.sub(lb_variables.index(i), ub_variables.index(i)),
                ),
                mf.Domain.equalsTo(polynomial[monomial]),
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
        a_sol = a.level()
        lb_sol = lb_variables.level()
        ub_sol = ub_variables.level()
        obj_sol = M.primalObjValue()
        computation_time = end_time - start_time

        solution = {"a": a_sol, "X": X_sol, "lb": lb_sol, "ub": ub_sol, "objective": obj_sol, "computation_time": computation_time}

    return solution


if __name__ == "__main__":
    seed = 1
    # Possible polynomials
    # ----------------------------------------
    # polynomial = polynomial_generation.Polynomial("x1^2 + x2^2 + 2x1x2", 2, 2)
    polynomial = polynomial_generation.Polynomial("normal_form", 4, 4, seed=seed)
    matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    matrix_size = len(matrix[0])

    # Solve unprojected unit sphere
    # ----------------------------------------
    a_sol, _ = solve_unprojected_unit_sphere(polynomial)
    print(" ")
    print("(CG paper approach)Results for a random normal norm of {} variables and degree {}".format(polynomial.n, polynomial.d))
    print("-" * 50)
    print("Original SDP obj:       {:.10f}".format(a_sol[0]))

    # # Solve dual unit sphere
    # # ----------------------------------------
    # y_sol, obj_sol = solve_dual_unit_sphere_polynomial_optimization_problem(polynomial)
    # print("Dual solution y :", y_sol)
    # print("Dual solution obj :", obj_sol)

    # Solve projected unit sphere
    # ----------------------------------------
    random_projector = random_projections.RandomProjector(
        round(matrix_size * 0.3), matrix_size, type="identity"
    )
    id_projected_solution = solve_projected_unit_sphere(polynomial, random_projector)
    print("Projected (id) SDP obj: {:.10f}".format(id_projected_solution["objective"]))
    print("-" * 50)

    print("Projected SDP obj:")
    print("Size     Value          Difference        Time")
    for rate in np.linspace(0.1, 1, 10):
        random_projector = random_projections.RandomProjector(
            round(matrix_size * rate), matrix_size, type="debug_constant", seed=seed
        )
        
        rp_solution = solve_projected_unit_sphere(
            polynomial, random_projector, verbose=False
        )
    
        increment = rp_solution["objective"] - a_sol[0] 
        # Print in table format with rate as column and then value and increment as other columns
        print("{:>3.2f}     {:>10.5f}     {:>10.5f}        {:>5.3f}".format(round(rate, 2), rp_solution["objective"], increment, rp_solution["computation_time"]))

    # print("")
    # print("Random projector used:")
    # monomials.print_readable_matrix(random_projector.projector)
