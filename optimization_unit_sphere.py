"""
This module contains functions to write the problem
of optimizing a polynomial function over the unit sphere
as a semidefinite program. 

minimize f(x)
subject to x^T x = 1

where f is a polynomial function.

We do this using teh standard SOS method. 

"""

import numpy as np
import polynomial_generation as poly
import monomials
import sys
import mosek.fusion as mf
import random_projections as rp
import time


def sum_tuples(t1, t2):
    """
    Sums two tuples element-wise.

    Parameters
    ----------
    t1 : tuple
        First tuple.
    t2 : tuple
        Second tuple.

    Returns
    -------
    tuple
        Sum of the two tuples.

    Examples
    --------
    >>> sum_tuples((1, 2, 3), (4, 5, 6))
    (5, 7, 9)

    >>> sum_tuples((1, 2, 3), (4, 5, 6, 7))
    (5, 7, 9, 7)

    """

    return tuple([t1[i] + t2[i] for i in range(min(len(t1), len(t2)))])


def add_tuple_to_tuple_list(tuple, list_of_tuples):
    """
    Sums a tuple to a list of tuples element-wise.

    Parameters
    ----------
    tuple : tuple
        Tuple.
    list_of_tuples : list
        List of tuples.

    Returns
    -------
    list
        List of tuples.

    Examples
    --------
    >>> add_tuple_to_tuple_list((1, 2, 3), [(4, 5, 6), (7, 8, 9)])
    [(5, 7, 9), (8, 10, 12)]

    >>> add_tuple_to_tuple_list((1, 2, 3), [(4, 5, 6), (7, 8, 9), (10, 11, 12, 13)])
    [(5, 7, 9), (8, 10, 12), (11, 13, 15, 13)]

    """

    return [sum_tuples(tuple, t) for t in list_of_tuples]


def sdp_relaxation(polynomial: poly.Polynomial, verbose=False):
    """
    Returns the SDP relaxation of the problem of optimizing a polynomial
    function over the unit sphere.

    max a
    s.t. f(x) - a = SOS(x) + (x^T x - 1) * POLY(x)

    max  b
    s.t. A_i · X + sum_j a_ij · w - a_s · w_j = c_i
         A_0 · X - w_0 + b = c_0
         X is psd

    where A_i is the coefficient picking of the i-th monomial in SOS(x)
    and a_(monomial) is the coefficient picking of the monomial of the
    sphere constraints.

    Parameters
    ----------
    polynomial : Polynomial
        Polynomial to be optimized.
    verbose : bool, optional

    Returns
    -------
    dict
        Dictionary with the solution of the SDP relaxation.

    """  

    monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)

    # Picking monomials from SOS polynomial
    A = {
        monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
        for monomial in distinct_monomials
    }

    # Picking monomials from the free polynomial (vector of coefficients)
    monomials_free_polynomial = monomials.generate_monomials_up_to_degree(
        polynomial.n, polynomial.d - 2
    )

    # Picking monomials from sphere constraint
    constraint_monomials = [
        tuple([0] * i + [2] + [0] * (polynomial.n - i - 1)) for i in range(polynomial.n)
    ]
    a = {}
    for i, monomial in enumerate(distinct_monomials):
        a[monomial] = {
            constraint_monomial: monomials.pick_specific_monomial(
                add_tuple_to_tuple_list(constraint_monomial, monomials_free_polynomial),
                monomial,
                vector=True,
            )
            for constraint_monomial in constraint_monomials
        }

    a_standard = {
        monomial: monomials.pick_specific_monomial(
            monomials_free_polynomial,
            monomial,
            vector=True,
        )
        for monomial in distinct_monomials
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Vector variable w
        w = M.variable(len(monomials_free_polynomial))

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_j a_j · w - a_standard w = c_i
        for monomial in [m for m in distinct_monomials if m != tuple_of_constant]:
            if verbose:
                print("A[{}]:".format(i + 1))
                monomials.print_readable_matrix(A[monomial])
                print("monomial: {}".format(monomial))
                print(
                    "polynomial coefficient: {}".format(polynomial.polynomial[monomial])
                )
                print("a[{}]:".format(i + 1))
                print(a[monomial])

            M.constraint(
                mf.Expr.sub(
                    mf.Expr.add(
                        mf.Expr.dot(A[monomial], X),
                        mf.Expr.add(
                            [
                                mf.Expr.dot(
                                    a[monomial][constraint_monomial],
                                    w,
                                )
                                for constraint_monomial in constraint_monomials
                            ]
                        ),
                    ),
                    mf.Expr.dot(
                        a_standard[monomial],
                        w,
                    ),
                ),
                mf.Domain.equalsTo(polynomial.polynomial[monomial]),
            )

        # Constraint: A_0 · X - w_0 + b = c_0
        if verbose:
            print("A[0]:")
            monomials.print_readable_matrix(A[tuple_of_constant])
            print("monomial: {}".format(tuple_of_constant))
            print(
                "polynomial coefficient: {}".format(
                    polynomial.polynomial[tuple_of_constant]
                )
            )

        M.constraint(
            mf.Expr.add(
                mf.Expr.sub(mf.Expr.dot(A[tuple_of_constant], X), w.index(0)), b
            ),
            mf.Domain.equalsTo(polynomial.polynomial[tuple_of_constant]),
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
        w_sol = w.level()
        computation_time = end_time - start_time

        if verbose:
            print("b_sol: {}".format(b_sol))
            print("w_sol: {}".format(w_sol))
            print("X_sol: {}".format(X_sol))

            print(
                "A_0 · X_sol = {}".format(np.dot(A[tuple_of_constant].flatten(), X_sol))
            )
            print("w_sol + b_sol = {}".format(w_sol + b_sol))
            print(
                "A_0 · X_sol - w_sol + b_sol = {}".format(
                    (np.dot(A[tuple_of_constant].flatten(), X_sol) - w_sol[0]) + b_sol
                )
            )
            print(
                bool(
                    np.dot(A[tuple_of_constant].flatten(), X_sol) - w_sol[0] - b_sol
                    == polynomial.polynomial[tuple_of_constant]
                )
            )

        solution = {
            "X": X_sol,
            "b": b_sol,
            "w": w_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
        }

        return solution


def projected_sdp_relaxation(
    polynomial: poly.Polynomial, random_projector: rp.RandomProjector, verbose=False
):
    """
    Returns the SDP relaxation of the problem of optimizing a polynomial
    function over the unit sphere.

    max  b + LB * sum(lbv) - UB * sum(ubv)
    s.t. PA_iP · X + sum_j a_ij · w - a_s · w_j + lbv[i] - ubv[i] = c_i
         PA_0P · X - w_0 + b + lbv[0] - ubv[0] = c_0
         X is psd
        lbv[i] >= 0, ubv[i] >= 0

    Parameters
    ----------
    polynomial : Polynomial
        Polynomial to be optimized.
    verbose : bool, optional

    Returns
    -------

    """

    old_monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d, old=True)

    raise SystemExit

    new_monomial_matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d, old=False)

    if not np.array_equal(old_monomial_matrix, new_monomial_matrix):
        print("The monomials matrix is different")
        print("old_monomial_matrix: {}".format(old_monomial_matrix))
        print("new_monomial_matrix: {}".format(new_monomial_matrix))
        print("The differences are:")
        print(np.setdiff1d(old_monomial_matrix, new_monomial_matrix))

    

    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    # print("distinct_monomials: {}".format(distinct_monomials))

    # Picking monomials from SOS polynomial
    # A = {
    #     monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
    #     for monomial in distinct_monomials
    # }

    A = {}
    for monomial in distinct_monomials:
        A[monomial] = random_projector.apply_rp_map(
            monomials.pick_specific_monomial(monomial_matrix, monomial)
        )

    # Picking monomials from the free polynomial (vector of coefficients)
    monomials_free_polynomial = monomials.generate_monomials_up_to_degree(
        polynomial.n, polynomial.d - 2
    )

    # Picking monomials from sphere constraint
    constraint_monomials = [
        tuple([0] * i + [2] + [0] * (polynomial.n - i - 1)) for i in range(polynomial.n)
    ]
    a = {}
    for i, monomial in enumerate(distinct_monomials):
        a[monomial] = {
            constraint_monomial: monomials.pick_specific_monomial(
                add_tuple_to_tuple_list(constraint_monomial, monomials_free_polynomial),
                monomial,
                vector=True,
            )
            for constraint_monomial in constraint_monomials
        }

    a_standard = {
        monomial: monomials.pick_specific_monomial(
            monomials_free_polynomial,
            monomial,
            vector=True,
        )
        for monomial in distinct_monomials
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Vector variable w
        w = M.variable(len(monomials_free_polynomial))

        # Lower and upper bounds
        lb_variables = M.variable(len(distinct_monomials) - 1, mf.Domain.greaterThan(0))
        ub_variables = M.variable(len(distinct_monomials) - 1, mf.Domain.greaterThan(0))

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = -1 - epsilon
        dual_upper_bound = 1 + epsilon

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Objective: maximize b + LB * sum(lbv) - UB * sum(ubv)
        b = M.variable()
        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.add(
                b,
                mf.Expr.sub(
                    mf.Expr.mul(
                        dual_lower_bound,
                        mf.Expr.dot(lb_variables, np.ones(len(distinct_monomials)-1)),
                    ),
                    mf.Expr.mul(
                        dual_upper_bound,
                        mf.Expr.dot(ub_variables, np.ones(len(distinct_monomials)-1)),
                    ),
                ),
            ),
        )

        # Constraints:
        # A_i · X + sum_j a_j · w - a_standard w + lbv[i] - ubv[i] = c_i
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
            print("monomial: {}".format(monomial))
            if verbose:
                print("A[{}]:".format(i + 1))
                monomials.print_readable_matrix(A[monomial])
                print("monomial: {}".format(monomial))
                print(
                    "polynomial coefficient: {}".format(polynomial.polynomial[monomial])
                )
                print("a[{}]:".format(i + 1))
                print(a[monomial])

            # matrix_inner_product = np.dot(random_projector.apply_rp_map(A[monomial]), X)
            matrix_inner_product = mf.Expr.dot(A[monomial], X)

            difference_slacks = mf.Expr.sub(
                lb_variables.index(i),
                ub_variables.index(i),
            )

            M.constraint(
                mf.Expr.add(
                    mf.Expr.sub(
                        mf.Expr.add(
                            matrix_inner_product,
                            mf.Expr.add(
                                [
                                    mf.Expr.dot(
                                        a[monomial][constraint_monomial],
                                        w,
                                    )
                                    for constraint_monomial in constraint_monomials
                                ]
                            ),
                        ),
                        mf.Expr.dot(
                            a_standard[monomial],
                            w,
                        ),
                    ),
                    difference_slacks,
                ),
                mf.Domain.equalsTo(polynomial.polynomial[monomial]),
            )

        # Constraint: A_0 · X - w_0 + b = c_0
        if verbose:
            print("A[0]:")
            monomials.print_readable_matrix(A[tuple_of_constant])
            print("monomial: {}".format(tuple_of_constant))
            print(
                "polynomial coefficient: {}".format(
                    polynomial.polynomial[tuple_of_constant]
                )
            )

        matrix_inner_product = mf.Expr.dot(A[tuple_of_constant], X)
        M.constraint(
                mf.Expr.add(mf.Expr.sub(matrix_inner_product, w.index(0)), b),
            mf.Domain.equalsTo(polynomial.polynomial[tuple_of_constant]),
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
        w_sol = w.level()
        lbv_sol = lb_variables.level()
        ubv_sol = ub_variables.level()

        computation_time = end_time - start_time

        if verbose:
            print("b_sol: {}".format(b_sol))
            print("w_sol: {}".format(w_sol))
            print("X_sol: {}".format(X_sol))
            print("lbv_sol: {}".format(lbv_sol))
            print("ubv_sol: {}".format(ubv_sol))

            print(
                "A_0 · X_sol = {}".format(np.dot(A[tuple_of_constant].flatten(), X_sol))
            )
            print("w_sol + b_sol = {}".format(w_sol + b_sol))
            print(
                "A_0 · X_sol - w_sol + b_sol = {}".format(
                    (np.dot(A[tuple_of_constant].flatten(), X_sol) - w_sol[0]) + b_sol
                )
            )
            print(
                bool(
                    np.dot(A[tuple_of_constant].flatten(), X_sol) - w_sol[0] - b_sol
                    == polynomial.polynomial[tuple_of_constant]
                )
            )

        solution = {
            "X": X_sol,
            "b": b_sol,
            "w": w_sol,
            "lbv": lbv_sol,
            "ubv": ubv_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
        }

        # print(
        #         "A_0 · X_sol = {}".format(np.dot(A[tuple_of_constant].flatten(), X_sol))
        #     )
        # print(
        #     "Size of X: {} x {}".format(
        #         A[distinct_monomials[0]].shape[0], A[distinct_monomials[0]].shape[0]
        #     )
        # )
        # print(
        #     "Rank of X: {}".format(
        #         np.linalg.matrix_rank(
        #             X_sol.reshape(
        #                 (
        #                     A[distinct_monomials[0]].shape[0],
        #                     A[distinct_monomials[0]].shape[0],
        #                 )
        #             )
        #         )
        #     )
        # )
        # print(
        #     "Eigenvalues of X: {}".format(
        #         np.linalg.eigvals(
        #             X_sol.reshape(
        #                 (
        #                     A[distinct_monomials[0]].shape[0],
        #                     A[distinct_monomials[0]].shape[0],
        #                 )
        #             )
        #         )
        #     )
        # )

    return solution


if __name__ == "__main__":
    seed = 1

    # Possible polynomials
    # ----------------------------------------
    # polynomial = polynomial_generation.Polynomial("x1^2 + x2^2 + 2x1x2", 2, 2)
    polynomial = poly.Polynomial("normal_form", 4, 4, seed=seed)
    # polynomial = poly.Polynomial("random", 4, 4, seed=seed)
    # polynomial = poly.Polynomial("3x1^2x2x3 + x1^2x2^2 + 10x3^4 + 5x2^2x1x3", 3, 4, seed=seed)
    matrix = monomials.generate_monomials_matrix(polynomial.n, polynomial.d)
    matrix_size = len(matrix[0])

    # Solve unprojected unit sphere
    # ----------------------------------------
    sdp_solution = sdp_relaxation(polynomial)
    print(" ")
    print(
        "Results for a random normal norm of {} variables and degree {}".format(
            polynomial.n, polynomial.d
        )
    )
    print("-" * 50)
    print("Original SDP:")
    print("Value         Computation time")
    print(
        "{:.10f}     {:.10f}".format(
            sdp_solution["objective"], sdp_solution["computation_time"]
        )
    )

    # Solve projected unit sphere
    # ----------------------------------------
    id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    id_rp_solution = projected_sdp_relaxation(polynomial, id_random_projector)
    print("Projected (id) SDP obj: {:.10f}".format(id_rp_solution["objective"]))
    print("-" * 50)

    print("Projected SDP obj:")
    print("Size     Value          Difference         Computation time")
    X_solutions = []
    for rate in np.linspace(0.6, 1, 5):
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type="sparse", seed=seed
        )

        rp_solution = projected_sdp_relaxation(
            polynomial, random_projector, verbose=False
        )

        increment = rp_solution["objective"] - sdp_solution["objective"]
        # Print in table format with rate as column and then value and increment as other columns
        print(
            "{:>3.2f}     {:>10.5f}     {:>10.5f}          {:>5.3f}".format(
                round(rate, 2),
                rp_solution["objective"],
                increment,
                rp_solution["computation_time"],
            )
        )
        X_solutions.append(rp_solution["X"])

        raise SystemExit

    print("-" * 50)
    for i, X in enumerate(X_solutions):
        print("Size of X_{}: {}".format(i, len(X)))
        print("X_{}: {}".format(i, X[:5]))
    print("-" * 50)
