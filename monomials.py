"""
This module contains functions related to monomials.

"""

import numpy as np
import math
import scipy
import itertools

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
    
    if n == 1:
        yield (d,)
    else:
        for value in range(d + 1):
            for permutation in generate_monomials_exact_degree(n - 1, d - value):
                yield (value,) + permutation
    

    
    
    # monomials = []
    # if d == 0:
    #     monomials.append(tuple(np.zeros(n)))
    # else:
    #     for i in range(n):
    #         for monomial in generate_monomials_exact_degree(n, d - 1):
    #             monomial = list(monomial)
    #             monomial[i] += 1
    #             monomials.append(tuple(monomial))

    # return monomials
    

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
        monomials += list(generate_monomials_exact_degree(n, i))
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


def pick_specific_monomial(monomials_matrix, monomial, vector=False):
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

    if vector:
        monomial_matrix = np.zeros(len(monomials_matrix))
        for i in range(len(monomials_matrix)):
            if monomials_matrix[i] == monomial:
                monomial_matrix[i] = 1

    else:
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


def generate_monomials_from_string(polynomial_string):
    """
    Generates a dictionary of monomials from a string representing a polynomial.

    Parameters
    ----------
    polynomial_string : string
        String representing a polynomial.

    Returns
    -------
    monomials : dict
        Dictionary of monomials.

    Examples
    --------
    >>> generate_monomials_from_string("2x1^2x2^2 + 3x1^2x2 + 4x1^2 + 5x1x2^2 + 6x1x2 + 7x1 + 8x2^2 + 9x2 + 10")
    {(2, 2): 2, (2, 1): 3, (2, 0): 4, (1, 2): 5, (1, 1): 6, (1, 0): 7, (0, 2): 8, (0, 1): 9, (0, 0): 10}

    """

    monomials = {}
    monomial_string_list = polynomial_string.split(" + ")
    for monomial_string in monomial_string_list:
        monomial, coefficient = parse_monomial(monomial_string)
        monomials[monomial] = coefficient

    return monomials


def parse_monomial(monomial_string, n):
    """
    Parses a string to a monomial.

    Parameters
    ----------
    monomial_string : string
        String representing a monomial.
    n : int
        Number of variables.

    Returns
    -------
    monomial : tuple
        Monomial represented by the string.
    coefficient : int
        Coefficient of the monomial.

    Examples
    --------
    >>> parse_monomial("2x1^2x2^2", 3)
    ((2, 2, 0), 2)

    """

    monomial = np.zeros(n)
    monomial_string_list = monomial_string.split("x")[1:]
    coefficient = monomial_string.split("x")[0]
    if coefficient == "":
        coefficient = 1
    else:
        coefficient = int(coefficient)

    for monomial_string in monomial_string_list:
        monomial_string = monomial_string.split("^")
        if len(monomial_string) == 1:
            monomial_string.append(1)
        monomial[int(monomial_string[0]) - 1] = int(monomial_string[1])

    return tuple(monomial), coefficient


def edge_to_monomial(edge, n):
    """
    Converts an edge to a monomial.

    Parameters
    ----------
    edge : tuple
        Edge of a graph.

    Returns
    -------
    monomial : tuple
        Monomial represented by the edge.

    Examples
    --------
    >>> edge_to_monomial((1, 2))
    (0, 1, 1)

    """

    monomial = np.zeros(n)
    monomial[edge[0]] = 1
    monomial[edge[1]] = 1

    return tuple(monomial)


def edges_to_monomials(edges, n):
    """
    Converts a list of edges to a list of monomials.

    Parameters
    ----------
    edges : list
        List of edges of a graph.

    Returns
    -------
    monomials : list
        List of monomials represented by the edges.

    Examples
    --------
    >>> edges_to_monomials([(1, 2), (2, 3)], 3)
    [(0, 1, 1), (0, 0, 1), (0, 0, 1)]

    """

    monomials = [edge_to_monomial(edge, n) for edge in edges]

    return monomials

def stable_set_monomial_matrix(edges, n, level=1):
    """
    Generates the second level monomial matrix for a specific graph.

    Main things are that powers of variables are replaced by 1.

    Parameters
    ----------
    edges : list
        List of edges of a graph.
    n : int
        Number of variables.

    Returns
    -------
    monomial_matrix : numpy.ndarray
        Matrix of monomials.

    """
    monomial_matrix = generate_monomials_matrix(n, 2 * level)
    new_matrix = []
    for row in monomial_matrix:
        new_row = []
        for tup in row:
            new_tup = tuple(1 if x in list(range(1, n)) else x for x in tup)
            new_row.append(new_tup)
        new_matrix.append(new_row)   
    
    monomial_matrix = new_matrix  

    return monomial_matrix 


def stable_set_distinct_monomials(edges, n, level=1):
    """
    Generates the second level distinct monomials for a specific graph.

    Main things are that powers of variables are not needed, same with 
    monomials involving edges.

    Parameters
    ----------
    edges : list
        List of edges of a graph.
    n : int
        Number of variables.

    Returns
    -------
    list_of_distinct_monomials : list
        List of distinct monomials.

    """

    distinct_monomials = generate_monomials_up_to_degree(n, 2 * level)
    new_vector = []

    for monomial in distinct_monomials:
        print('Filtering monomial {} out of {}'.format(monomial, len(distinct_monomials)))
        monomial_tuple = tuple(1 if x in list(range(1,n)) else x for x in monomial)
        contains_edge = False
        for edge in edges:
            tuple_edge = sum_tuples(monomial_tuple, edge_to_monomial(edge, n))
            if sum(x > 1 for x in tuple_edge) >= 2:
                contains_edge = True
                break 
        if not contains_edge:
            new_vector.append(monomial_tuple)
    
    # Remove repeated tuples
    unique_tuples = list(set(new_vector))
    distinct_monomials = unique_tuples
    
    return distinct_monomials