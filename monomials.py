"""
This module contains functions related to monomials.

"""

import numpy as np
import math
import scipy.special


def number_of_monomials(n, d):
    """
    Computes the number of monomials of a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.

    Returns
    -------
    number_of_monomials : int
        Number of monomials of degree d and dimension n.

    Examples
    --------
    >>> number_of_monomials(2, 2)
    3

    >>> number_of_monomials(3, 2)
    6

    """

    return scipy.special.comb(d + n - 1, n - 1, exact=True)


def number_of_monomials_up_to_degree(n, d):
    """
    Computes the number of monomials up to a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.

    Returns
    -------
    number_of_monomials : int
        Number of monomials of degree d and dimension n.

    Examples
    --------
    >>> number_of_monomials_up_to_degree(2, 2)
    6

    >>> number_of_monomials_up_to_degree(3, 2)
    10

    """

    return scipy.special.comb(d + n, n, exact=True)


def generate_monomials_exact_degree(n, d):
    """
    Generates all monomials of a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.
    string_representation : bool, optional
        If True, the monomials are returned as strings. The default is False.

    Returns
    -------
    monomials : list
        List of monomials of degree d and dimension n in tuple format.

    Examples
    --------
    >>> generate_monomials(2, 2)
    [(2, 0), (1, 1), (0, 2)]

    >>> generate_monomials(3, 2)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

    """

    monomials = []
    if d == 0:
        monomials.append(tuple(np.zeros(n)))
    else:
        for i in range(n):
            for monomial in generate_monomials_exact_degree(n, d - 1):
                monomial = list(monomial)
                monomial[i] += 1
                monomials.append(tuple(monomial))

    return monomials


def generate_monomials_up_to_degree(n, d):
    """
    Generates all monomials up to a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.

    Returns
    -------
    monomials : list
        List of monomials of degree d and dimension n in tuple format.

    Examples
    --------
    >>> generate_monomials_up_to_degree(2, 2)
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    >>> generate_monomials_up_to_degree(3, 2)
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

    """

    monomials = []
    for i in range(d + 1):
        monomials += generate_monomials_exact_degree(n, i)
    return monomials


def generate_monomials_matrix(n, d):
    """
    Generates matrix of monomials from outer product of vector of monomials
    of degree d/2 and n variables.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials in the matrix.

    Returns
    -------
    monomials_matrix : numpy.ndarray
        Matrix of monomials of degree d and dimension n.

    Examples
    --------
    >>> generate_monomials_matrix(2, 2)
    [
    [(0,0), (1, 0), (0, 1)],
    [(1,0), (2, 0), (1, 1)],
    [(0,1), (1, 1), (0, 2)]
    ]

    """

    monomials_vector = generate_monomials_up_to_degree(n, math.floor(d / 2))
    monomials_matrix = outer_product_monomials(monomials_vector)

    return monomials_matrix


def outer_product_monomials(monomials_vector):
    """
    Generates matrix of monomials from outer product of vector of monomials
    of degree d/2 and n variables.

    Parameters
    ----------
    monomials_vector : list
        Vector of monomials.

    Returns
    -------
    monomials_matrix : numpy.ndarray
        Matrix of monomials.

    Examples
    --------
    >>> outer_product_monomials([(0,0), (1, 0), (0, 1)])
    [
    [(0,0), (1, 0), (0, 1)],
    [(1,0), (2, 0), (1, 1)],
    [(0,1), (1, 1), (0, 2)]
    ]

    """

    monomials_matrix = []
    for i in range(len(monomials_vector)):
        monomials_matrix.append([])
        for j in range(len(monomials_vector)):
            monomials_matrix[i].append(
                tuple(
                    [
                        monomials_vector[i][k] + monomials_vector[j][k]
                        for k in range(len(monomials_vector[i]))
                    ]
                )
            )

    return monomials_matrix


def pick_specific_monomial(monomials_matrix, monomial):
    """
    Picks a specific monomial from the monomials matrix.
    Sets all entries to 0 except those corresponding to the monomial.

    Parameters
    ----------
    monomials_matrix : numpy.ndarray
        Matrix of monomials.
    monomial : tuple
        Monomial to be picked.

    Returns
    -------
    monomial_matrix : numpy.ndarray
        Matrix of monomial.

    Examples
    --------
    >>> pick_specific_monomial(generate_monomials_matrix(2, 2), (1, 1))
    [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
    ]
    """

    monomial_matrix = np.zeros((len(monomials_matrix), len(monomials_matrix)))
    for i in range(len(monomials_matrix)):
        for j in range(len(monomials_matrix)):
            if monomials_matrix[i][j] == monomial:
                monomial_matrix[i][j] = 1

    return monomial_matrix


def get_list_of_distinct_monomials(monomials_matrix):
    """
    Gets list of distinct monomials from the monomials matrix.

    Parameters
    ----------
    monomials_matrix : numpy.ndarray
        Matrix of monomials.

    Returns
    -------
    list_of_distinct_monomials : list
        List of distinct monomials.

    Examples
    --------
    >>> get_list_of_distinct_monomials(generate_monomials_matrix(2, 2))
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]

    """

    list_of_distinct_monomials = []
    for i in range(len(monomials_matrix)):
        for j in range(len(monomials_matrix)):
            if monomials_matrix[i][j] not in list_of_distinct_monomials:
                list_of_distinct_monomials.append(monomials_matrix[i][j])

    return list_of_distinct_monomials


def print_readable_matrix(monomials_matrix):
    """
    Prints matrix of monomials in readable format.

    Parameters
    ----------
    monomials_matrix : numpy.ndarray
        Matrix of monomials.

    Returns
    -------
    None.

    Examples
    --------
    >>> print_readable_matrix(generate_monomials_matrix(2, 2))
    (0, 0) (1, 0) (0, 1)
    (1, 0) (2, 0) (1, 1)
    (0, 1) (1, 1) (0, 2)

    """

    for i in range(len(monomials_matrix)):
        for j in range(len(monomials_matrix)):
            print(monomials_matrix[i][j], end=" ")
        print()
