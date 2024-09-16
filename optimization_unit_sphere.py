"""
This module contains functions to write the problem
of optimizing a polynomial function over the unit sphere
as a semidefinite program. 

minimize f(x)
subject to x^T x = 1

where f is a polynomial function.

We do this using the standard SOS method. 

"""

import numpy as np
import polynomial_generation as poly
import monomials
import sys
import mosek.fusion as mf
import random_projections as rp
import time
import scipy.special
import math
import scipy.linalg

def prod(iterable):
    """
    Returns the product of all the elements in the iterable.

    Parameters
    ----------
    iterable : iterable
        Iterable of numbers.

    Returns
    -------
    float
        Product of all the elements in the iterable.

    Examples
    --------
    >>> prod([1, 2, 3])
    6

    """

    result = 1
    for element in iterable:
        result *= element
    return result


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


# Traditional approach


def sdp_first_level_unit_sphere(polynomial: poly.Polynomial, verbose=False):
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

    distinct_monomials = polynomial.distinct_monomials

    # Picking monomials from SOS polynomial
    A = polynomial.A

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
        matrix_size = A[distinct_monomials[0]].shape[0]
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
        M.constraint(
            mf.Expr.add(
                mf.Expr.sub(mf.Expr.dot(A[tuple_of_constant], X), w.index(0)), b
            ),
            mf.Domain.equalsTo(polynomial.polynomial[tuple_of_constant]),
        )

        if verbose:
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        M.solve()
        end_time = time.time()

        # Get the solution

        solution = {
            "X": X.level().reshape(matrix_size, matrix_size),
            "b": b.level(),
            "w": w.level(),
            "objective": M.primalObjValue(),
            "size_psd_variable": matrix_size,
            "computation_time": end_time - start_time,
        }
        return solution


def projected_sdp_first_level_unit_sphere(
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

    distinct_monomials = polynomial.distinct_monomials

    A = {}
    for monomial in distinct_monomials:
        A[monomial] = random_projector.apply_rp_map(polynomial.A[monomial])

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
        matrix_size = A[distinct_monomials[0]].shape[0]
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Vector variable w
        w = M.variable(len(monomials_free_polynomial))

        # Lower and upper bounds
        lb_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))
        ub_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))

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
                        mf.Expr.dot(lb_variables, np.ones(len(distinct_monomials))),
                    ),
                    mf.Expr.mul(
                        dual_upper_bound,
                        mf.Expr.dot(ub_variables, np.ones(len(distinct_monomials))),
                    ),
                ),
            ),
        )

        # Constraints:
        # A_i · X + sum_j a_j · w - a_standard w + lbv[i] - ubv[i] = c_i
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

        # Constraint: A_0 · X - w_0 + b + lbv[i] - ubv[i] = c_0
        matrix_inner_product = mf.Expr.dot(A[tuple_of_constant], X)
        difference_slacks = mf.Expr.sub(lb_variables.index(0), ub_variables.index(0))
        M.constraint(
            mf.Expr.add(
                mf.Expr.add(mf.Expr.sub(matrix_inner_product, w.index(0)), b),
                difference_slacks,
            ),
            mf.Domain.equalsTo(polynomial.polynomial[tuple_of_constant]),
        )

        if verbose:
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        M.solve()
        end_time = time.time()

        solution = {
            "X": X.level().reshape(matrix_size, matrix_size),
            "b": b.level(),
            "w": w.level(),
            "lb": lb_variables.level(),
            "ub": ub_variables.level(),
            "objective": M.primalObjValue(),
            "size_psd_variable": matrix_size,
            "computation_time": end_time - start_time,
        }

    return solution


# CG approach


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
            polynomial[monomial] = scipy.special.factorial(d) / prod(
                [scipy.special.factorial(power / 2) for power in monomial]
            )

    return polynomial


def sdp_CG_unit_sphere(
    polynomial: poly.Polynomial,
    verbose: bool = False,
):
    """
    Solves a the first level of SOS relaxation of an polynomial optimization
    problem over the unit sphere using the method from the CG paper.

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

    """

    # Get list of all the matrices for picking coefficients.
    distinct_monomials = polynomial.distinct_monomials
    sphere_polynomial = get_sphere_polynomial(polynomial.n, polynomial.d / 2)

    A = polynomial.A

    # Get the coefficients of the original polynomial
    polynomial = polynomial.polynomial

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
        start_time = time.time()
        M.solve()
        end_time = time.time()

        solution = {
            "a": a.level(),
            # "X": X.level().reshape(m, m),
            "objective": M.primalObjValue(),
            "size_psd_variable": m,
            "no_constraints": len(distinct_monomials),
            "computation_time": end_time - start_time,
        }

    return solution


def sdp_dual_CG_unit_sphere(
    polynomial: poly.Polynomial,
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

        # Solve the problem
        M.solve()

        # Get the solution
        y_sol = y.level()
        obj_sol = M.primalObjValue()

    return y_sol, obj_sol


def projected_sdp_CG_unit_sphere(
    polynomial: poly.Polynomial,
    random_projector: rp.RandomProjector,
    verbose: bool = False,
):
    """
    Solves the problem of optimizing a polynomial over the unit sphere
    by projecting the matrices.

    Parameters
    ----------
    polynomial : poly.Polynomial
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
    distinct_monomials = polynomial.distinct_monomials
    sphere_polynomial = get_sphere_polynomial(polynomial.n, polynomial.d / 2)

    A = {}
    for i, monomial in enumerate(distinct_monomials):
        A[monomial] = random_projector.apply_rp_map(polynomial.A[monomial])

    # Get the coefficients of the original polynomial
    polynomial = polynomial.polynomial

    with mf.Model("SDP") as M:
        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        m = A[tuple_of_constant].shape[0]
        X = M.variable(mf.Domain.inPSDCone(m))
        a = M.variable()
        lb_variables = M.variable(
            "lb_variables", len(distinct_monomials), mf.Domain.greaterThan(0)
        )
        ub_variables = M.variable(
            "ub_variables", len(distinct_monomials), mf.Domain.greaterThan(0)
        )

        # Objective: (maximize) a + LB * sum(lbv) - UB * sum(ubv)
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

        # Constraints: A_i · X - a * s_i + lbv[i] - ubv[i] = c_i
        for i in range(len(A)):
            monomial = distinct_monomials[i]
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
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        M.solve()
        end_time = time.time()

        solution = {
            "a": a.level(),
            # "X": X.level().reshape(m, m),
            "lb": lb_variables.level(),
            "ub": ub_variables.level(),
            "objective": M.primalObjValue(),
            "size_psd_variable": m,
            "no_constraints": len(distinct_monomials),
            "computation_time": end_time - start_time,
        }

    return solution


def constraint_aggregation_CG_unit_sphere(
    polynomial: poly.Polynomial, random_projector: rp.RandomProjector, verbose=False
):
    """
    Solves a the constraint aggregation method of the unit sphere using the method
    from the CG paper.

    max a
    s.t. TA_i · X - a * Ts_i= Tc_i     i=0, ...
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

    """

    distinct_monomials = polynomial.distinct_monomials
    tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

    # Get the coefficients of the sphere polynomial
    sphere_polynomial_old = get_sphere_polynomial(polynomial.n, polynomial.d / 2)
    sphere_polynomial = {}
    # Aggregating the coefficients
    for i in range(random_projector.k):
        print(
            "Aggreating sphere coefficient {}/{}      ".format(i, random_projector.k),
            end="\r",
        )
        sphere_polynomial[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            sphere_polynomial[i] += (
                random_projector.projector[i, j] * sphere_polynomial_old[monomial]
            )

    # Get the matrices for picking coefficients.
    A_old = polynomial.A
    A = {}
    # Aggregating the matrices
    for i in range(random_projector.k):
        print(
            "Aggreating matrix {}/{}                      ".format(
                i, random_projector.k
            ),
            end="\r",
        )
        A[i] = np.zeros(
            (A_old[tuple_of_constant].shape[0], A_old[tuple_of_constant].shape[1])
        )
        for j, monomial in enumerate(distinct_monomials):
            A[i] += random_projector.projector[i, j] * A_old[monomial]

    # Get the coefficients of the original polynomial
    polynomial_old = polynomial.polynomial
    polynomial = {}
    # Aggregating the coefficients
    for i in range(random_projector.k):
        print(
            "Aggreating polynomial coefficient {}/{}".format(i, random_projector.k),
            end="\r",
        )
        polynomial[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            polynomial[i] += random_projector.projector[i, j] * polynomial_old[monomial]

    with mf.Model("SDP") as M:
        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        m = A[0].shape[0]

        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(m))

        # Objective: maximize a (scalar)
        a = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, a)

        # Constraint: A_i · X - a * s_i = c_i
        for i in range(random_projector.k):
            print(
                "Adding constraint {}/{}                  ".format(
                    i, random_projector.k
                ),
                end="\r",
            )
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(A[i], X),
                    mf.Expr.mul(sphere_polynomial[i], a),
                ),
                mf.Domain.equalsTo(polynomial[i]),
            )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        try:
            # Solve the problem
            start_time = time.time()
            M.solve()
            end_time = time.time()

            a = a.level()
            X = X.level().reshape(m, m)
            objective = M.primalObjValue()
            computational_time = end_time - start_time

        except:
            a = math.inf
            objective = np.inf
            computational_time = np.inf

        solution = {
            "a": a,
            "X": X,
            "objective": objective,
            "size_psd_variable": m,
            "no_constraints": random_projector.k,
            "computation_time": computational_time,
        }

    return solution


def combined_projection_CG_unit_sphere(
    polynomial: poly.Polynomial,
    random_projector_variables: rp.RandomProjector,
    random_projector_constraints: rp.RandomProjector,
    verbose=False,
):
    """
    Solves a the constraint aggregation method of the unit sphere using the method
    from the CG paper.

    max a
    s.t. TPA_iP · Y - a * Ts_i= Tc_i     i=0, ...
         Y is positive semidefinite

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

    """

    distinct_monomials = polynomial.distinct_monomials
    tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

    # Get the coefficients of the sphere polynomial
    sphere_polynomial_old = get_sphere_polynomial(polynomial.n, polynomial.d / 2)
    sphere_polynomial = {}
    # Aggregating the coefficients
    for i in range(random_projector_constraints.k):
        print(
            "Aggreating sphere coefficient {}/{}      ".format(
                i, random_projector_constraints.k
            ),
            end="\r",
        )
        sphere_polynomial[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            sphere_polynomial[i] += (
                random_projector_constraints.projector[i, j]
                * sphere_polynomial_old[monomial]
            )

    # Get the matrices for picking coefficients.
    A_old = {}
    for i, monomial in enumerate(distinct_monomials):
        A_old[monomial] = random_projector_variables.apply_rp_map(
            polynomial.A[monomial]
        )
    A = {}
    # Aggregating the matrices
    for i in range(random_projector_constraints.k):
        print(
            "Aggreating matrix {}/{}                      ".format(
                i, random_projector_constraints.k
            ),
            end="\r",
        )
        A[i] = np.zeros(
            (A_old[tuple_of_constant].shape[0], A_old[tuple_of_constant].shape[1])
        )
        for j, monomial in enumerate(distinct_monomials):
            A[i] += random_projector_constraints.projector[i, j] * A_old[monomial]

    # Get the coefficients of the original polynomial
    polynomial_old = polynomial.polynomial
    polynomial = {}
    # Aggregating the coefficients
    for i in range(random_projector_constraints.k):
        print(
            "Aggreating polynomial coefficient {}/{}".format(
                i, random_projector_constraints.k
            ),
            end="\r",
        )
        polynomial[i] = 0
        for j, monomial in enumerate(distinct_monomials):
            polynomial[i] += (
                random_projector_constraints.projector[i, j] * polynomial_old[monomial]
            )

    with mf.Model("SDP") as M:
        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])
        m = A[0].shape[0]

        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(m))

        # Objective: maximize a (scalar)
        a = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, a)

        # Constraint: A_i · X - a * s_i = c_i
        for i in range(random_projector_constraints.k):
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(A[i], X),
                    mf.Expr.mul(sphere_polynomial[i], a),
                ),
                mf.Domain.equalsTo(polynomial[i]),
            )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        try:
            # Solve the problem
            start_time = time.time()
            M.solve()
            end_time = time.time()

            a = a.level()
            X = X.level().reshape(m, m)
            objective = M.primalObjValue()
            computational_time = end_time - start_time

        except:
            a = math.inf
            objective = math.inf
            computational_time = math.inf

        solution = {
            "a": a,
            "X": X,
            "objective": objective,
            "size_psd_variable": A_old[tuple_of_constant].shape[0],
            "no_constraints": random_projector_constraints.k,
            "computation_time": computational_time,
        }

    return solution


def single_polynomial_table(
    polynomial,
    type_variable,
    type_constraints,
    range,
    iterations, 
    form=True,
    seed=0,
):
    """
    Get the results for a single polynomial.

    Parameters
    ----------
    polynomial : Polynomial
        Polynomial to be optimized.
    projection_variable : string
        Type of projection for the variables.
    projection_constraints : string
        Type of projection for the constraints.
    range_variable : list
        Range of the projection for the variables.
    range_constraints : list
        Range of the projection for the constraints.

    Returns
    -------
    None

    """

    # Solve unprojected CG unit sphere
    # ----------------------------------------
    print("\n" + "-" * 80)
    print(
        "Results for a polynomial with {} variables and degree {}".format(
            polynomial.n, polynomial.d
        ).center(80)
    )
    print("-" * 80)
    print("\n{: <18} {: >10} {: >8} {: >8}".format("Type", "Size X", "Value", "Time"))
    print("-" * 80)

    if form:
        CG_sdp_solution = sdp_CG_unit_sphere(polynomial, verbose=False)
        print(
            "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                "CG original SDP",
                CG_sdp_solution["size_psd_variable"],
                CG_sdp_solution["objective"],
                CG_sdp_solution["computation_time"],
            )
        )

    else:
        print("This method requires a form polynomial.")

    print("- " * 40)

    # # Solve unprojected unit sphere
    # # ----------------------------------------
    # sdp_solution = sdp_first_level_unit_sphere(polynomial, verbose=False)
    # print(
    #     "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
    #         "L1 original SDP",
    #         sdp_solution["size_psd_variable"],
    #         sdp_solution["objective"],
    #         sdp_solution["computation_time"],
    #     )
    # )
    # print("- " * 40)

    # Solve projected CG unit sphere
    # ----------------------------------------
    matrix_size = CG_sdp_solution["size_psd_variable"]
    for rate in np.linspace(range[0], range[1], iterations):
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type_variable
        )
        if form:
            CG_rp_solution = projected_sdp_CG_unit_sphere(
                polynomial, random_projector, verbose=False
            )
            print(
                "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                    "CG projection " + str(round(rate, 2)),
                    CG_rp_solution["size_psd_variable"],
                    CG_rp_solution["objective"],
                    CG_rp_solution["computation_time"],
                )
            )

        else:
            print("This method requires a form polynomial.")

    #     # rp_solution = projected_sdp_first_level_unit_sphere(
    #     #     polynomial, random_projector, verbose=False
    #     # )
    #     # print(
    #     #     "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
    #     #         "L1 projection " + str(round(rate, 2)),
    #     #         rp_solution["size_psd_variable"],
    #     #         rp_solution["objective"],
    #     #         rp_solution["computation_time"],
    #     #     )
    #     # )

    print("- " * 40)

    # Solve constraint aggregation CG unit sphere
    # ----------------------------------------
    no_constraints = len(polynomial.distinct_monomials)
    for rate in np.linspace(range[0], range[1], iterations):
        random_projector = rp.RandomProjector(
            round(no_constraints * rate), no_constraints, type=type_constraints
        )
        if form:
            CG_rp_solution = constraint_aggregation_CG_unit_sphere(
                polynomial, random_projector, verbose=False
            )
            print(
                "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                    "CG aggregation " + str(round(rate, 2)),
                    CG_rp_solution["size_psd_variable"],
                    CG_rp_solution["objective"],
                    CG_rp_solution["computation_time"],
                )
            )
        else:
            print("This method requires a form polynomial.")

    print("- " * 40)

    # Solve combined projection CG unit sphere
    # ----------------------------------------
    no_constraints = len(polynomial.distinct_monomials)
    matrix_size = CG_sdp_solution["size_psd_variable"]
    for rate in np.linspace(range[0], range[1], iterations):
        random_projector_variables = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type_variable
        )
        random_projector_constraints = rp.RandomProjector(
            round(no_constraints * rate), no_constraints, type=type_constraints
        )
        if form:
            CG_rp_solution = combined_projection_CG_unit_sphere(
                polynomial,
                random_projector_variables,
                random_projector_constraints,
                verbose=False,
            )
            print(
                "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
                    "CG combined " + str(round(rate, 2)),
                    CG_rp_solution["size_psd_variable"],
                    CG_rp_solution["objective"],
                    CG_rp_solution["computation_time"],
                )
            )
        else:
            print("This method requires a form polynomial.")

    print("- " * 40)

    # # Solve projected unit sphere with identity projector
    # # ----------------------------------------
    # id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    # id_rp_solution = projected_sdp_first_level_unit_sphere(
    #     polynomial, id_random_projector, verbose=False
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

    print("No. distinct monomials (constraints): ", len(polynomial.distinct_monomials))


def alternating_projection_sdp(which_affine, polynomial, random_projector, solution_matrix, objective_feasible):
    """
    Alternating projection method to alternate between the
    SDP solution found in the constraint aggregation/combined 
    projection and the original and projected affine subspace.

    The projection to affine subspace A ⊙ X = b is:
    X = X' + A^T (AA^T)^-1 (b − A ⊙ X')

    Parameters
    ----------
    which_affine : string
        Which affine subspace to project onto.
    polynomial : Polynomial
        Polynomial to be optimized.
    random_projector : RandomProjector
        Random projector we used to project the variables.
    solution_matrix : numpy.ndarray
        Solution of the constraint aggregation.
    objective_feasible : float
        Feasible objective value of the variable reduction. 

    """

    iter = 0
    psd_X = solution_matrix
    tuple_of_constant = tuple([0 for i in list(polynomial.A.keys())[0]])
    
    # Constraints are: A_i · X - a * s_i + lbv[i] - ubv[i] = c_i
    # So we project onto the affine subspaces  A_i · X = c_i + a * s_i (we need an a that we know is feasible)
    sphere_polynomial = get_sphere_polynomial(polynomial.n, polynomial.d / 2)
    sphere_polynomial = {key: value * objective_feasible for key, value in sphere_polynomial.items()}
    rhs = {key: sphere_polynomial[key] + polynomial.polynomial[key] for key in sphere_polynomial.keys()}
    
    if which_affine == "original":
        A_i = polynomial.A 
    elif which_affine == "projected":
        A_i = {}
        for monomial in polynomial.distinct_monomials:
            A_i[monomial] = random_projector.apply_rp_map(polynomial.A[monomial])
    
    while iter < 1000:
        print("Iteration: ", iter)
        # Check if linear constraints are satisfied
        for monomial in A_i.keys():
            if np.all(np.dot(A_i[monomial], psd_X) == rhs[monomial]):
                print(
                    "Linear constraints satisfied for psd matrix at iteration {}".format(
                        iter
                    )
                )
                return psd_X, rhs[tuple_of_constant] - psd_X[0, 0]

        # Project onto the original affine subspace with orthogonal projection
        A = np.array([A_i[monomial].flatten() for monomial in A_i.keys()])
        inverse_AA = np.linalg.inv(A @ A.T)
        b_AX = np.array(
            [rhs[monomial] - np.trace(A_i[monomial].T @ psd_X) for monomial in A_i.keys()]
        )
        ATAATbAX = A.T @ inverse_AA @ b_AX
        affine_X = psd_X + ATAATbAX.reshape(psd_X.shape)
        affine_X.reshape(psd_X.shape)

        # Check if the projection is psd
        eigenvalues = np.linalg.eigvals(affine_X)
        if np.all(eigenvalues >= -0.0001):
            print("Projection onto affine subspace is psd at iteration {}".format(iter))
            return affine_X, rhs[tuple_of_constant] - affine_X[0, 0]
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

        print(rhs[tuple_of_constant] - psd_X[0, 0]) 
    return psd_X, rhs[tuple_of_constant] - psd_X[0, 0]

if __name__ == "__main__":
    seed = 1
    # Possible polynomials
    # ----------------------------------------
    # polynomial = poly.Polynomial("x1^2 + x2^2 + 2x1x2", 2, 2)
    # polynomial = poly.Polynomial("random", 15, 4, seed=seed)
    polynomial = poly.Polynomial("normal_form", 10, 4, seed=seed)

    # Run the table
    # ----------------------------------------
    # single_polynomial_table(polynomial, "0.5_density", "0.5_density", [0.9, 0.9], 1, form=True)

    # # Run the alternating projection
    # Solve original SDP
    # ----------------------------------------
    CG_sdp_solution = sdp_CG_unit_sphere(polynomial, verbose=False)
    # Solve variable projected SDP
    # ----------------------------------------
    random_projector = rp.RandomProjector(round(0.9 * CG_sdp_solution["size_psd_variable"]), CG_sdp_solution["size_psd_variable"], type="0.2_density")
    CG_rp_solution = projected_sdp_CG_unit_sphere(polynomial, random_projector, verbose=False)
    # Solve constraint aggregation
    # ----------------------------------------
    random_projector_constraints = rp.RandomProjector(round(0.9 * len(polynomial.distinct_monomials)), len(polynomial.distinct_monomials), type="0.2_density")
    CG_ca_solution = constraint_aggregation_CG_unit_sphere(polynomial, random_projector_constraints, verbose=False)

    # Alternating projection
    # ----------------------------------------
    solution_matrix = CG_ca_solution["X"]
    objective_feasible = CG_rp_solution["objective"]
    psd_X, objective = alternating_projection_sdp("original", polynomial, random_projector, CG_ca_solution["X"], objective_feasible)

    print("SDP objective: ", CG_sdp_solution["objective"])
    print("Objective constraint aggregation: ", CG_ca_solution["objective"])
    print("Objective value after APM to projected: ", objective)


