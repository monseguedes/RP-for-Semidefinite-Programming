"""
Module to solve simple uncontrained polynomial optimization problem 
using Mosek. 

Author: Monse Guedes Ayala
"""

import numpy as np
import scipy.special
import monomials
import polynomial_generation
import mosek
import mosek.fusion as mf
import sys

variables = 2
degree = 2

polynomial = polynomial_generation.Polynomial('x1^2 + x2^2 + 2x1x2', 2, 2)
random_polynomial = polynomial_generation.Polynomial('random', variables, degree)
print('Polynomial:', random_polynomial.polynomial)

def solve_unconstrained_polynomial_optimization_problem(polynomial: polynomial_generation.Polynomial, relaxation_degree):
    """
    Solves a the relaxation of an unconstrained polynomial optimization problem using Mosek.

    max g 
    s.t. A_i · X = c_i
         A_0 · X - g = c_0
         X is positive semidefinite

    where X is a symmetric matrix, A_i are symmetric matrices that pick coefficients, 
    and c_i are the coefficients of f(x).

    Parameters
    ----------
    polynomial : numpy.ndarray
        Polynomial to be optimized.
    relaxation_degree : int
        Degree of the relaxation.

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

    polynomial = polynomial.polynomial

    b = polynomial.values()
    b = list(b)
    print('b:', b)
    print('Length of b:', len(b))

    A = []
    for monomial in distinct_monomials:
        A_i = monomials.pick_specific_monomial(monomial_matrix, monomial)
        # monomials.print_readable_matrix(A_i)
        A.append(A_i)
    print('Length of A:', len(A))

    with mf.Model("SDP") as M:
        # Variable X is a symmetric matrix of size (r.degree + n choose n)
        n = A[0].shape[0]
        X = M.variable(mf.Domain.inPSDCone(n))

        # Constraint: Inner product of A_i and X should be equal to b_i for all i
        for i in range(1, len(A)):
            print('A[{}]:'.format(i))
            monomials.print_readable_matrix(A[i])
            print('monmial: {}'.format(distinct_monomials[i]))
            M.constraint(mf.Expr.dot(A[i], X), mf.Domain.equalsTo(b[i]))

        # Objective: Maximize a (scalar)
        a = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, a)
        print('A[0]:')
        monomials.print_readable_matrix(A[0])
        print('monmial: {}'.format(distinct_monomials[0]))

        M.constraint(mf.Expr.sub(mf.Expr.dot(A[1], X), a), mf.Domain.equalsTo(b[1]))

        # Constraint: PSD constraint on X
        M.constraint(X, mf.Domain.inPSDCone())

        # Solve the problem
        M.solve()

        # Get the solution
        X_sol = X.level()
        a_sol = a.level()

    return a_sol, X_sol

# Example usage:
a_sol, X_sol = solve_unconstrained_polynomial_optimization_problem(random_polynomial, 2)

print('Done')
print('Solution:', a_sol)





