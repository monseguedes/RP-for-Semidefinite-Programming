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
    s.t. A_i · X + sum_(monomials in sphere) a_(monomial) · w - w_0 = c_i
         A_0 · X - w_0 - b = c_0
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

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Vector variable w
        w = M.variable(len(monomials_free_polynomial))

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraint: A_i · X + sum_(monomials in sphere) a_(monomial) · w - w_0 = c_i
        for i, monomial in enumerate(
            [m for m in distinct_monomials if m != tuple_of_constant]
        ):
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
                                mf.Expr.dot(a[monomial][constraint_monomial], w)
                                for constraint_monomial in constraint_monomials
                            ]
                        )
                    ),
                    w.index(0)
                ),
                mf.Domain.equalsTo(polynomial.polynomial[monomial]),
            )

        # Constraint: A_0 · X - w_0 - b = c_0
        print("A[0]:")
        monomials.print_readable_matrix(A[tuple_of_constant])
        print("monomial: {}".format(tuple_of_constant))
        print("polynomial coefficient: {}".format(polynomial.polynomial[tuple_of_constant]))
    
        M.constraint(
            mf.Expr.sub(mf.Expr.sub(mf.Expr.dot(A[tuple_of_constant], X), 
                                    w.index(0)), 
                        b),
            mf.Domain.equalsTo(polynomial.polynomial[tuple_of_constant]),
        )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        # Solve the problem
        M.solve()

        # Get the solution
        X_sol = X.level()
        a_sol = a.level()


if __name__ == "__main__":
    polynomial = poly.Polynomial("normal_form", 4, 2, seed=0)
    polynomial = poly.Polynomial("x1^2 + x2^2 + 2x1x2", 2, 2)
    print(polynomial.polynomial)
    sdp_relaxation(polynomial, verbose=True)
