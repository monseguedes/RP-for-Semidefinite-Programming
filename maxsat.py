"""
This script contains the implementation of the MAX-2-SAT problem
using the SDP relaxation method and its projection. 

The script also contains the implementation of the random projection
map for SDP problems, i.e. m_P(A):A --> PAP^T, where P is a random
projection matrix. We solve the following problem:

    max <W, Y>
    s.t. Y is PSD
        Y_ii = 1 for all i

Notation convension:
For a for formula F = (¬x1 v x2) ^ (x3 v ¬x4) ^ ... ^ (x2 v x1), we
represent it as a list of tuples, where each tuple represents a clause
in the formula. For example, the formula 

F = (¬x1 v x2) ^ (x3 v ¬x4) ^ (x2 v x1)

is represented as:

F = [(-1, 2), (3, -4), (2, 1)]

where each tuple represents a clause in the formula.

The objective function is the sum of the clauses in the formula. For
example, the objective function for the formula F is:

sum v(c) for c in F

where v(c) = 1 if c is true, and v(c) = 0 otherwise.

We introduce a variable y_i ∈ {-1, 1} for each variable xi in the formula, 
and and additional y_0 ∈ {-1, 1}. We define x_i to be true iff y_i = y_0.

Therefore, v(x_i) = 1/2(1 + y_i * y_0), and v(¬x_i) = 1/2(1 - y_i * y_0).

Following this, v(x_i v x_j) = 1/4 (1 + y_i * y_0) + (1 + y_j * y_0) + (1 - y_i *  y_j), 

If a variable x_i is negated, we replace y_i with -y_i in the above expression.

The objective function is then the sum of the clauses in the formula, and
the problem is to maximize the objective function.

From a formula F with n variables, we define the corresponding y_i, y_0, and
calculate the coefficients of the objective function. 

"""

import numpy as np
import networkx as nx
import mosek.fusion as mf
import time
import matplotlib.pyplot as plt
import pickle
import random_projections as rp
from process_graphs import File
import math
import random

class Formula():
    def __init__(self, n, c, list_of_clauses=[], seed=0):
        self.seed = seed
        self.n = n
        self.c = c
        print("Generating formula...".ljust(80), end="\r")
        self.list_of_clauses = self.generate_formula(list_of_clauses)
        self.y_pairs = self.get_all_y_pairs()
        self.coefficients = self.get_all_coefficients()
        self.coefficients_matrix = self.get_coefficients_matrix()
        print("Formula generated.".ljust(80), end="\r")

    def generate_formula(self, list_of_clauses):
        """
        Generate a random clause for the formula.

        Parameters
        ----------
        args : list
            A list of tuples representing clauses

        Returns
        -------
        tuple
            A tuple representing a clause in the formula.
        """

        np.random.seed(self.seed)

        # Check that all tuples are orderes from smaller to larger index        
        for clause in list_of_clauses:
            if abs(clause[0]) > abs(clause[1]):
                raise ValueError("The variables in the clause are not ordered from smaller to larger index.")
            if abs(clause[0]) == abs(clause[1]):
                raise ValueError("Two variables in the clause are the same.")
            if abs(clause[0]) > self.n or abs(clause[1]) > self.n:
                raise ValueError("The variables in the clause are greater than the number of variables in the formula.")
        
        if list_of_clauses != []:
            list_of_clauses = list_of_clauses

        else:
            list_of_clauses = []
            while len(list_of_clauses) < self.c:
                x_i = random.choice([i for i in range(-self.n, self.n + 1) if i != 0 and i != self.n])
                x_j = random.choice([-1, 1]) * random.choice([i for i in range(abs(x_i), self.n + 1) if i != 0])
                if (x_i, x_j) not in list_of_clauses and abs(x_i) != abs(x_j):
                    list_of_clauses.append((x_i, x_j))

        return list_of_clauses
    
    def get_all_y_pairs(self):
        """
        Get all possible combinations of the variables in the formula.
        These are expressed as all tuple combinations of y_0 and y_i.

        Returns
        -------
        list
            List of all possible combinations of the variables in the formula.
        """

        y_pairs = [(0,0)] + [(i, j) for i in range(self.n + 1) for j in range(i + 1, self.n + 1)]

        return y_pairs
    
    def get_coefficients_clause(self, clause):
        """
        Get the coefficients of the vlaue of a clause in the formula.

        Parameters
        ----------
        clause : tuple
            A tuple representing a clause in the formula.

        Returns
        -------
        dict
            A dictionary representing the coefficients of the clause.
        """

        coefficients = {}
        coefficients[(0,0)] = 3 * 1/4
        for x_i in clause:
            if x_i < 0:
                coefficients[(0, -x_i)] = 1/4
            else:
                coefficients[(0, x_i)] = 1/4

        coefficients[(abs(clause[0]), abs(clause[1]))] = - 1/4 * np.sign(clause[0]) * np.sign(clause[1])

        return coefficients
    
    def get_all_coefficients(self):
        """
        Add the coefficients of the objective function to the y pairs.

        Returns
        -------
        list
            List of tuples representing the coefficients of the objective function.
        """

        coefficients = {
            pair : 0 for pair in self.y_pairs
        }

        for clause in self.list_of_clauses:
            clause_coefficients = self.get_coefficients_clause(clause)
            for key, value in clause_coefficients.items():
                coefficients[key] += value

        return coefficients
    
    def get_coefficients_matrix(self):
        """
        Returns the symmetric matrix of coefficients of the objective function.

        Returns
        -------
        numpy.ndarray
            The matrix of coefficients of the objective function.
        """

        matrix = np.zeros((self.n + 1, self.n + 1))

        for key, value in self.coefficients.items():
            if key[0] !=  key[1]:
                matrix[key[0], key[1]] = 1/2 * value
                matrix[key[1], key[0]] = 1/2 * value
            elif key[0] == 0 and key[1] == 0:
                matrix[key[0], key[1]] = value

        return matrix


def sdp_relaxation(formula: Formula):
    """
    Solves the MaxCut problem using the SDP relaxation
    method.

    This problem is formulated as a semidefinite program
    as follows:

    max <W, X>
    s.t. X is PSD
         X_ii = 1 for all i
    
    where W is the matrix of coefficients of the objective.

    Parameters
    ----------
    formula : Formula
        The formula representing the MAX-2-SAT problem.

    Returns
    -------
    numpy.ndarray
        The solution to the MaxCut problem.
    """

    W = formula.coefficients_matrix

    with mf.Model("SDP") as M:
        # PSD variable X
        matrix_size = formula.n + 1
        X = M.variable(mf.Domain.inPSDCone(matrix_size))

        # Objective:
        M.objective(mf.ObjectiveSense.Maximize, mf.Expr.dot(W, X))

        # Constraints:
        constraints = []
        for i in range(matrix_size):
            print("Adding constraints... {}/{}              ".format(i + 1, matrix_size), end="\r")
            constraints.append(M.constraint(X.index(i, i), mf.Domain.equalsTo(1)))

        start_time = time.time()
        print("Solving original SDP...".ljust(80), end="\r")
        M.solve()
        print("Solved")
        end_time = time.time()


        solution = {
            # "X_sol": X.level().reshape((matrix_size, matrix_size)),
            "objective": M.primalObjValue(),
            "size_psd_variable": matrix_size,
            "computation_time": end_time - start_time,
            "C": formula.c / formula.n,
            "n": formula.n,
        }

        return solution


def projected_sdp_relaxation(formula, projector, verbose=False, slack=True):
    """ 
    Solves the MAX-2-SAT problem projecting the SDP relaxation
    using a random projector.

    This problem is formulated as a semidefinite program
    as follows:

    max <PWP^T, X>
    s.t. X is PSD
         <PAP^T, X> = 1 for all A in A
        
    where W is the matrix of coefficients of the objective, and
    A are the matrices that pick entry ii in the matrix.

    Parameters
    ----------
    formula : Formula
        The formula representing the MAX-2-SAT problem.
    projector : RandomProjector
        The random projector to be used.
    verbose : bool, optional
        Whether to print the progress of the algorithm. Defaults to False.
    slack : bool, optional
        Whether to add slack variables to the constraints. Defaults to True.

    """

    W = formula.coefficients_matrix
    original_dimension = W.shape[0]
    W = projector.apply_rp_map(W)
    n = W.shape[0]

    A_matrix = np.zeros((n, n))
    A = {}
    for i in range(original_dimension):
        print("Creating A matrix... {}/{}".format(i + 1, original_dimension), end="\r")
        A_matrix = np.zeros((original_dimension, original_dimension))
        A_matrix[i, i] = 1
        A[i] = A_matrix

    # projected_A = {}
    # for i in range(original_dimension):
    #     print("Projecting A matrix... {}/{}".format(i + 1, original_dimension), end="\r")
    #     projected_A[i] = projector.apply_rp_map(A[i])

    projected_A = {}
    for i in range(original_dimension):
        print("Projecting A matrix... {}/{}".format(i + 1, original_dimension), end="\r")
        projected_A[i] = np.outer(projector.projector[:, i], projector.projector[:, i])

    with mf.Model("SDP") as M:
        # PSD variable X
        X = M.variable(mf.Domain.inPSDCone(n))
        lb_variables = M.variable(original_dimension, mf.Domain.greaterThan(0))
        ub_variables = M.variable(original_dimension, mf.Domain.greaterThan(0))
        ones_vector = np.ones(original_dimension)

        # Lower and upper bounds of the dual variables
        epsilon = 0.00001
        dual_lower_bound = -1000000 - epsilon
        dual_upper_bound = 1000000 + epsilon

        # Objective:
        # M.objective(mf.ObjectiveSense.Maximize, mf.Expr.mul(1 / 4, mf.Expr.dot(L, X)))
        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.add(
                mf.Expr.dot(W, X),
                mf.Expr.sub(
                    mf.Expr.mul(
                        dual_lower_bound,
                        mf.Expr.dot(lb_variables, ones_vector),
                    ),
                    mf.Expr.mul(
                        dual_upper_bound,
                        mf.Expr.dot(ub_variables, ones_vector),
                    ),
                ),
            )
        )

        # Constraints:
        # constraints = []
        for i in range(original_dimension):
            print("Adding constraints... {}/{}              ".format(i + 1, original_dimension), end="\r")
            difference_slacks = mf.Expr.sub(
                lb_variables.index(i),
                ub_variables.index(i),
            )
            M.constraint(
                    mf.Expr.add(mf.Expr.dot(projected_A[i], X), difference_slacks),
                    mf.Domain.equalsTo(1),
                )
        

        start_time = time.time()
        print("Solving projected SDP with {} variables and projector {}".format(projector.k, projector.type).ljust(80))
        M.solve()
        print("Solved")
        end_time = time.time()

        solution = {
            # "X_sol": X.level().reshape((n, n)),
            "objective": M.primalObjValue(),
            "size_psd_variable": n,
            "computation_time": end_time - start_time,
        }

        return solution


def satisfiability_feasibility(formula: Formula):
    """
    Solves the 2-SAT staisfiability problem using the SDP relaxation
    method.

    This problem is formulated as a semidefinite program
    as follows:
    find X
    s.t. X is PSD
         X_ii = 1 for all i
         sign(x) * X_0x + sign(y) * X_0y - sign(xy) * X_xy = 1 for all clauses (x v y)

    Parameters
    ----------
    formula : Formula
        The formula representing the MAX-2-SAT problem.

    Returns
    -------
    numpy.ndarray
        The solution to the 2-SAT problem.
    """

    clauses = formula.list_of_clauses

    with mf.Model("SDP") as M:
        # PSD variable X
        matrix_size = formula.n + 1
        X = M.variable(mf.Domain.inPSDCone(matrix_size))

        # Constant objective:
        M.objective(mf.ObjectiveSense.Maximize, 1)

        # Constraints:
        constraints = []
        # Diagonal equal to 1
        # for i in range(matrix_size):
        #     print("Adding constraints... {}/{}              ".format(i + 1, matrix_size), end="\r")
        #     constraints.append(M.constraint(X.index(i, i), mf.Domain.equalsTo(1)))
        # Alternative for diagonal
        for i in range(matrix_size):
            matrix = np.zeros((matrix_size, matrix_size))
            matrix[i, i] = 1
            constraints.append(M.constraint(mf.Expr.dot(matrix, X), mf.Domain.equalsTo(1)))
        # Clauses
        for clause in clauses:
            matrix = np.zeros((matrix_size, matrix_size))
            # First clause
            matrix[abs(clause[0]), 0] = np.sign(clause[0])
            matrix[0, abs(clause[0])] = np.sign(clause[0])
            # Second clause
            matrix[0, abs(clause[1])] = np.sign(clause[1])
            matrix[abs(clause[1]), 0] = np.sign(clause[1])
            # Combined 
            matrix[abs(clause[0]), abs(clause[1])] = -1 * np.sign(clause[0]) * np.sign(clause[1])
            matrix[abs(clause[1]), abs(clause[0])] = -1 * np.sign(clause[0]) * np.sign(clause[1])
            constraints.append(M.constraint(mf.Expr.dot(matrix, X), mf.Domain.equalsTo(1)))

        start_time = time.time()
        try:                
            M.solve()
            end_time = time.time()
            objective = M.primalObjValue()
        except:
            objective = 0
            end_time = time.time()                   
        

        solution = {
            "objective": objective,
            "size_psd_variable": matrix_size,
            "computation_time": end_time - start_time,
            "C": formula.c / formula.n,
            "n": formula.n,
        }

        return solution
    
def projected_sat_feasibility(formula, projector):
    """ 
    Solves the 2-SAT problem projecting the SDP relaxation
    using a random projector.

    This problem is formulated as a semidefinite program
    as follows:
    find X
    s.t. X is PSD
         X_ii = 1 for all i PROJECTING THE CONSTRAINTS
         sign(x) * X_0x + sign(y) * X_0y - sign(xy) * X_xy = 1 for all clauses (x v y) PROJECTING THE CONSTRAINTS

    Parameters
    ----------
    formula : Formula
        The formula representing the MAX-2-SAT problem.
    projector : RandomProjector
        The random projector to be used.
    verbose : bool, optional
        Whether to print the progress of the algorithm. Defaults to False.
    slack : bool, optional
        Whether to add slack variables to the constraints. Defaults to True.

    """

    clauses = formula.list_of_clauses

    with mf.Model("SDP") as M:
        # PSD variable X
        matrix_size = formula.n + 1
        projected_size = projector.k
        X = M.variable(mf.Domain.inPSDCone(matrix_size))

        # Constant objective:
        M.objective(mf.ObjectiveSense.Maximize, 1)

        # Constraints:
        constraints = []
        # Diagonal equal to 1
        for i in range(matrix_size):
            matrix = np.zeros((matrix_size, matrix_size))
            matrix[i, i] = 1
            matrix = projector.apply_rp_map(matrix)
            constraints.append(M.constraint(mf.Expr.dot(matrix, X), mf.Domain.equalsTo(1)))
        # Clauses
        for i, clause in enumerate(clauses):
            # print("Adding clause constraints... {}/{}              ".format(i + 1, len(clauses), end="\r"))
            matrix = np.zeros((matrix_size, matrix_size))
            # First clause
            matrix[abs(clause[0]), 0] = np.sign(clause[0])
            matrix[0, abs(clause[0])] = np.sign(clause[0])
            # Second clause
            matrix[0, abs(clause[1])] = np.sign(clause[1])
            matrix[abs(clause[1]), 0] = np.sign(clause[1])
            # Combined 
            matrix[abs(clause[0]), abs(clause[1])] = -1 * np.sign(clause[0]) * np.sign(clause[1])
            matrix[abs(clause[1]), abs(clause[0])] = -1 * np.sign(clause[0]) * np.sign(clause[1])
            matrix = projector.apply_rp_map(matrix)
            constraints.append(M.constraint(mf.Expr.dot(matrix, X), mf.Domain.equalsTo(1)))

        start_time = time.time()
        try:                 
            M.solve()
            end_time = time.time()
            objective = M.primalObjValue()
        except:
            objective = 0
            end_time = time.time()


        solution = {
            "objective": objective,
            "size_psd_variable": projected_size,
            "computation_time": end_time - start_time,
            "C": formula.c / formula.n,
            "n": formula.n,
        }

        return solution
    


def single_formula_results(formula, type="sparse", range=(0.1, 0.5), iterations=5, problem="max"):
    """
    Get the results for a single graph.

    Parameters
    ----------
    graph : Graph
        Graph object representing the input graph.
    type : str, optional
        Type of random projector. Defaults to "sparse".
    range : tuple, optional
        Range of rates for the random projector. Defaults to (0.1, 0.5).
    iterations : int, optional
        Number of iterations for the random projector. Defaults to 5.
    """

    # Solve unprojected stable set problem
    # ----------------------------------------
    print("\n" + "-" * 80)
    print("Results for a formula with {} variables and {} clauses".format(formula.n, formula.c).center(80))
    print("-" * 80)
    print(
        "\n{: <18} {: >10} {: >8} {: >12} {: >8} ".format(
            "Type", "Size X", "Value", "Quality/SDP", "Time", 
        )
    )
    print("-" * 80)

    if problem == "max":
        sdp_solution = sdp_relaxation(formula)
    elif problem == "sat":
        sdp_solution = satisfiability_feasibility(formula)
    print(
        "{: <18} {: >10} {: >8.2f} {: >12} {: >8.2f}".format(
            "SDP Relaxation",
            sdp_solution["size_psd_variable"],
            sdp_solution["objective"],
            "-",
            sdp_solution["computation_time"],
        )
    )

    matrix_size = sdp_solution["size_psd_variable"]

    for rate in np.linspace(range[0], range[1], iterations):
        slack = True
        if rate > 0.5:
            slack = True
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type
        )
        if problem == "max":
            rp_solution = projected_sdp_relaxation(
                formula, random_projector, verbose=False, slack=slack
            )
            quality = rp_solution["objective"] / sdp_solution["objective"] * 100
            quality = str(round(quality, 2)) + "%"
        elif problem == "sat":
            rp_solution = projected_sat_feasibility(formula, random_projector)
            quality = "N/A"
       
        print(
            "{: <18} {: >10} {: >8.2f} {: >12} {: >8.2f}".format(
                "Projection " + str(round(rate, 2)),
                rp_solution["size_psd_variable"],
                rp_solution["objective"],
                quality,
                rp_solution["computation_time"],
            )
        )

    # # Solve identity projector
    # # ----------------------------------------
    # id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    # id_rp_solution = projected_sdp_relaxation(
    #     formula, id_random_projector, verbose=False, slack=False
    # )
    # quality = id_rp_solution["objective"] / sdp_solution["objective"] * 100
    # print(
    #     "{: <18} {: >10} {: >8.2f} {:>12} {: >8.2f}".format(
    #         "Identity",
    #         id_rp_solution["size_psd_variable"],
    #         id_rp_solution["objective"],
    #         str(round(quality, 2)) + "%",
    #         id_rp_solution["computation_time"],
    #     )
    # )

    print()



if __name__ == "__main__":
    # Create a formula
    variables = 500
    C = 2
    formula = Formula(variables, variables * C)
    # print(formula.list_of_clauses)

    # single_formula_results(formula, type="sparse", range=(0.1, 0.9), iterations=9)
    # satisfiability_feasibility(formula)
    # projector = rp.RandomProjector(50, 101, type="sparse")
    # projected_sat_feasibility(formula, projector)
    single_formula_results(formula, type="0.1_density", range=(0.1, 0.3), iterations=3, problem="sat")
