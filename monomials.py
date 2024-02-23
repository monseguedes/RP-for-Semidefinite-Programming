"""
This module contains functions related to monomials.

"""

import numpy as np
import math
import scipy
import itertools
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

    # if n == 1:
    #     yield (d,)
    # else:
    #     for value in range(d + 1):
    #         for permutation in generate_monomials_exact_degree(n - 1, d - value):
    #             yield (value,) + permutation

    if n == 1:
        yield (d,)
    else:
        for value in range(d + 1):
            for permutation in generate_monomials_exact_degree(n - 1, d - value):
                yield permutation + (value,) 




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


def generate_monomials_matrix(n, d, stable_set=False):
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
    monomials_matrix = outer_product_monomials(monomials_vector, stable_set=stable_set)

    return monomials_matrix


def outer_product_monomials(monomials_vector, stable_set=False):
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
            if stable_set:
                monomials_matrix[i].append(
                    tuple(
                        [
                            int(monomials_vector[i][k] + monomials_vector[j][k] != 0)
                            for k in range(len(monomials_vector[i]))
                        ]
                    )
                )
            else:
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
    monomial_matrix = generate_monomials_matrix(n, 2 * level, stable_set=True)

    # new_matrix = []
    # for row in monomial_matrix:
    #     new_row = []
    #     for tup in row:
    #         new_tup = tuple(1 if x in list(range(1, n)) else x for x in tup)
    #         new_row.append(new_tup)
    #     new_matrix.append(new_row)

    # monomial_matrix = new_matrix

    return monomial_matrix


def generate_tuples(n, d):
    """
    Generates all tuples of length n with d ones and
    all the rest 0s.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Number of ones.

    Returns
    -------
    tuples : list
        List of tuples.

    Examples
    --------
    >>> generate_tuples(3, 2)
    [(0,1,1), (1,0,1), (1,1,0)]

    """

    def generate_helper(curr_tuple, remaining_ones, remaining_zeros):
        if len(curr_tuple) == n:
            print(curr_tuple)
        else:
            if remaining_ones > 0:
                generate_helper(curr_tuple + [1], remaining_ones - 1, remaining_zeros)
            if remaining_zeros > 0:
                generate_helper(curr_tuple + [0], remaining_ones, remaining_zeros - 1)

    generate_helper([], n - d, d)


def generate_binary_monomials_exact_degree(n, d):
    """
    Generates all binary monomials of a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.

    Returns
    -------
    monomials : list
        List of binary monomials of degree d and dimension n in tuple format.

    Examples
    --------
    >>> generate_binary_monomials_exact_degree(2, 2)
    [(1, 1)]

    >>> generate_binary_monomials_exact_degree(3, 2)
    [(0, 1, 1), (1, 1, 0), (1, 0, 1)]

    """

    if n == 1:
        yield (min(d, 1),)
    else:
        for value in range(min(d, 1) + 1):
            for permutation in generate_binary_monomials_exact_degree(n - 1, d - value):
                yield (value,) + permutation


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

    if level == 1:
        distinct_monomials = []
        print("Generating distinct monomials level 1 degree 1...")
        degree_1 = generate_binary_monomials_exact_degree(n, 1)
        print("Generating distinct monomials level 1 degree 2...")
        degree_2 = generate_binary_monomials_exact_degree(n, 2)
        print("Filtering distinct monomials level 1 degree 2...")
        degree_2 = [monomial for monomial in degree_2 if monomial not in edges_to_monomials(edges, n)]
        # # Filter those in edges
        # filtered_degree_2 = []
        # no_monomials = len(list(degree_2))
        # for i, monomial in enumerate(list(degree_2)):
        #     print("Filtering monomial {} out of {}".format(i, no_monomials))
        #     if not any(
        #         sum(1 for x, y in zip(monomial, edge) if x == 1 and y == 1) >= 2
        #         for edge in edges_to_monomials(edges, n)
        #     ):
        #         filtered_degree_2.append(monomial)

                
        distinct_monomials = list(set(list(degree_1) + degree_2 + [tuple(0 for _ in range(n))]))

    if level == 2:
        # #OLD METHOD
        # start = time.time()
        # distinct_monomials = generate_monomials_up_to_degree(n, 2 * level)
        # new_vector = []

        # for i,  monomial in enumerate(distinct_monomials):
        #     # print('Filtering monomial {} out of {}'.format(i, len(distinct_monomials)))
        #     monomial_tuple = tuple(1 if x in list(range(1,n)) else x for x in monomial)
        #     contains_edge = False
        #     for edge in edges:
        #         tuple_edge = sum_tuples(monomial_tuple, edge_to_monomial(edge, n))
        #         if sum(x > 1 for x in tuple_edge) >= 2:
        #             contains_edge = True
        #             break
        #     if not contains_edge:
        #         new_vector.append(monomial_tuple)

        # # Remove repeated tuples
        # unique_tuples = list(set(new_vector))
        # distinct_monomials = unique_tuples
        # end = time.time()
        # print('Time elapsed distinct monomials old method: {}'.format(end - start))
        # print('Number of distinct monomials: {}'.format(len(distinct_monomials)))

        # NEW METHOD
        print("Generating distinct monomials level 2 degree 2...")
        start = time.time()
        distinct_monomials = []
        degree_1 = generate_binary_monomials_exact_degree(n, 1)
        degree_2 = generate_binary_monomials_exact_degree(n, 2)
        degree_2 = [
            monomial
            for monomial in degree_2
            if not any(
                sum(1 for x, y in zip(monomial, edge) if x == 1 and y == 1) >= 2
                for edge in edges_to_monomials(edges, n)
            )
        ]
        print("Generating distinct monomials level 2 degree 3...")
        degree_3 = generate_binary_monomials_exact_degree(n, 3)
        degree_3 = [
            monomial
            for monomial in degree_3
            if not any(
                sum(1 for x, y in zip(monomial, edge) if x == 1 and y == 1) >= 2
                for edge in edges_to_monomials(edges, n)
            )
        ]
        print("Generating distinct monomials level 2 degree 4...")
        degree_4 = generate_binary_monomials_exact_degree(n, 4)
        degree_4 = [
            monomial
            for monomial in degree_4
            if not any(
                sum(1 for x, y in zip(monomial, edge) if x == 1 and y == 1) >= 2
                for edge in edges_to_monomials(edges, n)
            )
        ]
        distinct_monomials = list(
            set(
                list(degree_1)
                + degree_2
                + degree_3
                + degree_4
                + [tuple(0 for _ in range(n))]
            )
        )
        end = time.time()
        print("Time elapsed distinct monomials new method: {}".format(end - start))
        print("Number of distinct monomials: {}".format(len(distinct_monomials)))

    return distinct_monomials
