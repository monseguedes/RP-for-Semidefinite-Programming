"""
This module contains functions to write the polynomial
optimization problem for the stable set problem.

minimize    sum x_v
subject to  x_u * x_v = 0 for all (u,v) in E
            x_v^2 = x_v for all v in V

where E is the set of edges and V is the set of vertices.

We do this using the SOS method, i.e. we solve the problem

minimize    a
subject to  sum x_v - a = SOS + sum POLY_j (x_v * x_u) + sum POLY_k (x_v^2 - x_v)

"""

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import monomials
import sys
import mosek.fusion as mf
import time


def get_list_of_edges(graph, full=False):
    """
    Get the list of edges of the graph.

    Parameters
    ----------
    graph : matrix
        Adjacency matrix of the graph.

    Returns
    -------
    list
        List of edges of the graph.
    """

    n = graph.shape[0]
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i, j] == 1:
                edges.append((i, j))

    if full:
        edges_dict = {}
        for i in range(n):
            for j in range(i + 1, n):
                edges_dict[(i, j)] = 0
        for edge in edges:
            edges_dict[edge] = 1

        return edges_dict

    return edges


def generate_graph(n, p, seed=0):
    """
    Generate a random graph with n vertices and edge probability p.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Probability of an edge between any two vertices.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    matrix
        Adjacency matrix of the graph.
    """

    # Set the random seed
    np.random.seed(seed)

    # Generate random values for edges
    edges = np.random.rand(n, n)

    # Create a mask for edges based on probability
    graph = (edges < p).astype(int)

    # Make sure the diagonal is zero (no self-loops)
    np.fill_diagonal(graph, 0)

    # Make the matrix symmetric
    graph = np.triu(graph) + np.triu(graph, 1).T

    return graph


class Graph:
    def __init__(self, n, p, seed=0):
        self.n = n
        self.p = p
        self.graph = generate_graph(self.n, self.p, seed)
        self.edges = get_list_of_edges(self.graph)

    def plot_graph(self):
        """
        Plot the graph.

        Parameters
        ----------
        graph : matrix
            Adjacency matrix of the graph.

        """

        G = nx.Graph(self.graph)
        nx.draw(G, with_labels=True, font_weight="bold")
        plt.show()


def stable_set_mip(graph):
    raise NotImplementedError


def stable_set_problem_sdp(graph: Graph, verbose=False):
    """
    Write the polynomial optimization problem for the stable set problem.

    minimize    a
    subject to  sum x_v - a = SOS + sum POLY_j (x_v * x_u) + sum POLY_k (x_v^2 - x_v)

    which can be written as

    minimize    a
    subject to  A_i · X + sum_j∈E B_ij · POL_j(x_u * x_v) + sum_v∈V C_iv · POL_v(x_v2) - D_iv · POLv(x_v) = c_i
                A_0 · X + sumj∈E B_0j · POL_j(x_u * x_v) + sum_v∈V C_0v · POL_v(x_v2) - D_0v · POLv(x_v) = c_0

    where E is the set of edges and V is the set of vertices.

    NOTE: WE ARE ONLY SOLVING THE FIRST LEVEL OF THE HIERARCHY, WHICH MEANS THAT
    ALL POL CAN BE AT MOST DEGREE 1.

    Parameters
    ----------
    graph : Graph
        Graph object.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    monomial_matrix = monomials.generate_monomials_matrix(graph.n, 2)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    degree_1_monomials = monomials.generate_monomials_exact_degree(graph.n, 1)
    degree_2_monomials = [
        monomial
        for monomial in monomials.generate_monomials_exact_degree(graph.n, 2)
        if any(n == 2 for n in monomial)
    ]

    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Picking monomials from SOS polynomial
    A = {
        monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_(u,v) (x_u * x_v)
    E = {
        monomial: monomials.pick_specific_monomial(
            monomials.edges_to_monomials(edges, graph.n), monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v^2)
    V_squared = {
        monomial: monomials.pick_specific_monomial(
            degree_2_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v)
    V = {
        monomial: monomials.pick_specific_monomial(
            degree_1_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Constant for (x_v * x_u)
        e = M.variable(len(graph.edges), mf.Domain.unbounded())

        # Constant for (x_v^2 - x_v)
        v = M.variable(graph.n, mf.Domain.unbounded())

        # Objective: maximize a (scalar)
        b = M.variable()
        M.objective(mf.ObjectiveSense.Maximize, b)

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v = c_i
        for monomial in [m for m in distinct_monomials if m != tuple_of_constant]:
            M.constraint(
                mf.Expr.add(
                    mf.Expr.dot(A[monomial], X),
                    mf.Expr.add(
                        mf.Expr.dot(E[monomial], e),
                        mf.Expr.sub(
                            mf.Expr.dot(V_squared[monomial], v),
                            mf.Expr.dot(V[monomial], v),
                        ),
                    ),
                ),
                mf.Domain.equalsTo(C[monomial]),
            )

        # Constraint:
        # A_0 · X + b = c_0
        M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
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
        computation_time = end_time - start_time

        solution = {
            "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
        }

        return solution


def projected_stable_set_problem_sdp(graph: Graph, verbose=False):
    """
    Write the projected problem for the stable set problem.

    minimize    a
    subject to  A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v
                + lbv[i] - ubv[i] = c_i
                A_0 · X + b + lbv[0] - ubv[0] = c_0

    where E is the set of edges and V is the set of vertices.

    NOTE: WE ARE ONLY SOLVING THE FIRST LEVEL OF THE HIERARCHY, WHICH MEANS THAT
    ALL POL CAN BE AT MOST DEGREE 1.

    Parameters
    ----------
    graph : matrix or list
        Adjacency matrix of the graph or list of edges.

    Returns
    -------
    dict
        Dictionary with the solutions of the sdp relaxation.

    """

    monomial_matrix = monomials.generate_monomials_matrix(graph.n, 2)
    distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
    degree_1_monomials = monomials.generate_monomials_exact_degree(graph.n, 1)
    degree_2_monomials = [
        monomial
        for monomial in monomials.generate_monomials_exact_degree(graph.n, 2)
        if any(n == 2 for n in monomial)
    ]
    edges = graph.edges

    # Coefficients of objective
    C = {monomial: -1 if sum(monomial) == 1 else 0 for monomial in distinct_monomials}

    # Picking monomials from SOS polynomial
    A = {
        monomial: monomials.pick_specific_monomial(monomial_matrix, monomial)
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_(u,v) (x_u * x_v)
    E = {
        monomial: monomials.pick_specific_monomial(
            monomials.edges_to_monomials(edges, graph.n), monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v^2)
    V_squared = {
        monomial: monomials.pick_specific_monomial(
            degree_2_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    # Picking monomials for POLY_v (x_v)
    V = {
        monomial: monomials.pick_specific_monomial(
            degree_1_monomials, monomial, vector=True
        )
        for monomial in distinct_monomials
    }

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(A[distinct_monomials[0]].shape[0]))

        # Constant for (x_v * x_u)
        e = M.variable(len(graph.edges), mf.Domain.unbounded())

        # Constant for (x_v^2 - x_v)
        v = M.variable(graph.n, mf.Domain.unbounded())

        # Lower and upper bounds
        lb_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))
        ub_variables = M.variable(len(distinct_monomials), mf.Domain.greaterThan(0))

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = 0 - epsilon
        dual_upper_bound = 1 + epsilon

        # Objective: maximize a (scalar)
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

        tuple_of_constant = tuple([0 for i in range(len(distinct_monomials[0]))])

        # Constraints:
        # A_i · X + sum_e E · constant_e + V_squared · constant_v - V · constrant_v = c_i
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
                    matrix_inner_product,
                    mf.Expr.add(
                        mf.Expr.dot(E[monomial], e),
                        mf.Expr.sub(
                            mf.Expr.dot(V_squared[monomial], v),
                            mf.Expr.dot(V[monomial], v),
                        ),
                    ),
                ),
                mf.Domain.equalsTo(C[monomial]),
            )

        # Constraint:
        # A_0 · X + b = c_0
        M.constraint(
            mf.Expr.add(mf.Expr.dot(A[tuple_of_constant], X), b),
            mf.Domain.equalsTo(C[tuple_of_constant]),
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
        computation_time = end_time - start_time

        solution = {
            "X": X_sol,
            "b": b_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
        }

        return solution


graph = Graph(5, 0.5)
graph.plot_graph()
solution = stable_set_problem_sdp(graph)
print(solution["objective"])
