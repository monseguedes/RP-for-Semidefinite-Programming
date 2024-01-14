"""
Module for generating polynomials either randomly, from a given set of coefficients
and variables, or from a string.

"""

import numpy as np
import monomials


def generate_random_polynomial(n, d, seed=0, density=0.5):
    """
    Generates a random polynomial of a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the polynomial.
    seed : int
        Seed for the random generator.

    Returns
    -------
    polynomial : numpy.ndarray
        Random polynomial of degree d and dimension n.

    Examples
    --------
    >>> generate_random_polynomial(2, 2)
    array([  0,  34,   0, -14,   0,   0])

    """

    polynomial = {
        tuple: 0
        for tuple in monomials.get_list_of_distinct_monomials(
            monomials.generate_monomials_matrix(n, d)
        )
    }

    np.random.seed(seed)
    coefficients = np.random.randint(
        -100, 100, size=monomials.number_of_monomials_up_to_degree(n, d)
    )
    coefficients[
        np.random.choice(len(polynomial), int(len(polynomial) * density), replace=False)
    ] = 0

    for i, monomial in enumerate(polynomial):
        polynomial[monomial] = coefficients[i]

    return polynomial


def parse_polynomial(polynomial_string, n, d):
    """
    Parses a string to a polynomial.

    Parameters
    ----------
    polynomial_string : string
        String representing a polynomial.
    n : int
        Number of variables.

    Returns
    -------
    polynomial : dict
        Dictionary representing a polynomial with monomials and their coefficients.
        Ordered by monomial.

    Examples
    --------
    >>> parse_polynomial("2x1^2x2^2 + 3x1^2x2 + 4x1x2^2 + 5x1x2 + 6x1 + 7x2 + 8", 2)
    {(2, 2): 2, (2, 1): 3, (1, 2): 4, (1, 1): 5, (1, 0): 6, (0, 1): 7, (0, 0): 8}
    """

    polynomial = {
        tuple: 0
        for tuple in monomials.get_list_of_distinct_monomials(
            monomials.generate_monomials_matrix(n, d)
        )
    }
    monomial_string_list = polynomial_string.split(" + ")
    for monomial_string in monomial_string_list:
        monomial, coefficient = monomials.parse_monomial(monomial_string, n)
        polynomial[monomial] = coefficient

    return polynomial
