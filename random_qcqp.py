"""
Generate random QCQP problems taken from paper

Nonconvex quadratically constrained quadratic
programming: best D.C. decompositions and their SDP
representations by X. J. Zheng, X. L. Sun, D. Li.

Here we consider the following problem:

minimize    f(x) = x^T A_0 x + b_0^T x
subject to  g(x) = x^T A_i x + b_i^T x + c_i <= 0 for i=1,...,m
            Dx <= d, 
            0 <= x_i <= 1 for i=1,...,n

where D is a lxn matrix, and d is a l-dimensional vector.

The matrices A_i take the form A_i = P_i T_i P_i^T, for some orthogonal
matrix P_i and diagonal matrix T_i. 

    P_i = Q_1 Q_2 Q_3 where Q_j = I - 2 w_j w_j^T / ||w_j||^2 j=1,2,3

and components of w_j are chosen uniformly at random from (-1,1).

T_i are as follows:
    - T_0 = Diag(T_01, ..., T_0n) with T_0i ∈ (-5n, 0) for i=1,...,floor(n/2)
    and T_0i ∈ (0, 5n) for i=floor(n/2)+1,...,n
    - T_i = Diag(T_i1, ..., T_in) with T_ij ∈ (0, 5n) for j=1,...,m

The vector b_0 = (b_01, ..., b_0n)^T where b_0j ∈ (-100, 100), and the
vectors b_i = (b_i1, ..., b_in)^T where b_ij ∈ (-50n, 0) for i=1,...,m.

Lastly, c_i ∈ (-50, 0) for i=1,...,m.

The parameters in the linear constraints are chosen as follows:
    - D = (Dij) with Dij ∈ (0, 50) 
    - d = De/n where e is the vector of all ones.

The SDP relaxation of this problem is given by:

minimize    <M_0, Z>
subject to  <M_i, Z> <= 0 for i=1,...,m
            <L, Z> <= 0 
            <R_i, Z> - 1 <=0 for i=0,...,n
            - <R_i, Z> <= 0 for i=0,...,n
            Z => 0
            Z = [[X, x], [x^T, 1]]

            
where M_i = [[A_i, b_i],[b_i^T, c_i]], L = [[D, 0], [0, 0]], and
R_i = 1/2 * [[0, ..., 0, 1],[0, ..., 0, 1]].

"""

import numpy as np
import math
import mosek.fusion as mf
import time
import sys
import random_projections as rp


def build_A_i(i, n):
    """
    Build A_i = P_i T_i P_i^T

    where P_i = Q_1 Q_2 Q_3 and Q_j = I - 2 w_j w_j^T / ||w_j||^2 j=1,2,3
    and components of w_j are chosen uniformly at random from (-1,1).

    T_i are as follows:
        - T_0 = Diag(T_01, ..., T_0n) with T_0i ∈ (-5n, 0) for i=1,...,floor(n/2)
        and T_0i ∈ (0, 5n) for i=floor(n/2)+1,...,n
        - T_i = Diag(T_i1, ..., T_in) with T_ij ∈ (0, 5n) for j=1,...,m

    Parameters
    ----------
    i : int
        The index of the A_i matrix.
    n : int
        The size of the A_i matrix.

    Returns
    -------
    numpy.ndarray
        The A_i matrix.

    """

    # Build P_i
    w_1 = np.random.uniform(-1, 1, n)
    Q_1 = np.eye(n) - 2 * np.outer(w_1, w_1) / np.linalg.norm(w_1) ** 2
    w_2 = np.random.uniform(-1, 1, n)
    Q_2 = np.eye(n) - 2 * np.outer(w_2, w_2) / np.linalg.norm(w_2) ** 2
    w_3 = np.random.uniform(-1, 1, n)
    Q_3 = np.eye(n) - 2 * np.outer(w_3, w_3) / np.linalg.norm(w_3) ** 2

    P_i = Q_1 @ Q_2 @ Q_3

    if i == 0:
        T_i = np.zeros((n, n))
        for i in range(math.floor(n / 2)):
            T_i[i, i] = np.random.uniform(-0.1 * n, 0) # -5 n
        for i in range(math.floor(n / 2) + 1, n):
            T_i[i, i] = np.random.uniform(0, 0.1 * n) # 5 n

    else:
        T_i = np.diag(np.random.uniform(0, 0.1 * n, n)) # 5 n 

    A_i = P_i @ T_i @ P_i.T

    return A_i


def build_b_i(i, n):
    """
    Build the vector b_i = (b_i1, ..., b_in)^T where b_ij ∈ (-50n, 0) for i=1,...,m.

    If i=0, then b_0j ∈ (-100, 100).

    Parameters
    ----------
    i : int
        The index of the b_i vector.
    n : int
        The size of the b_i vector.

    Returns
    -------
    numpy.ndarray
        The b_i vector.

    """

    if i == 0:
        b_i = np.random.uniform(-0.1, 0.1, n)  # 100

    b_i = np.random.uniform(-0.5 * n, 0, n)  # 50

    return b_i


def build_c_i(i):
    """
    Build the vector c_i ∈ (-50, 0) for i=1,...,m.

    Parameters
    ----------
    m : int
        The number of c_i vectors.

    Returns
    -------
    numpy.ndarray
        The c_i vector.

    """

    if i == 0:
        c_i = 0

    else:
        c_i = np.random.uniform(-0.5, 0)  # 50

    return c_i


def build_D(l, n):
    """
    Build the matrix D = (Dij) with Dij ∈ (0, 50).

    Parameters
    ----------
    l : int
        The number of rows of the D matrix.
    n : int
        The number of columns of the D matrix.

    Returns
    -------
    numpy.ndarray
        The D matrix.

    """

    D = np.random.uniform(0, 5, (l, n))  # 50

    return D


def build_d(D, l, n):
    """
    Build the vector d = De/n where e is the vector of all ones.

    Parameters
    ----------
    l : int
        The size of the d vector.
    n : int
        The size of the d vector.

    Returns
    -------
    numpy.ndarray
        The d vector.

    """

    d = D @ np.ones(n) / n

    return d


def build_M_i(A_i, b_i, c_i):
    """
    Build the matrix M_i = [[A_i, b_i],[b_i^T, c_i]].

    Parameters
    ----------
    i : int
        The index of the M_i matrix.
    n : int
        The size of the A_i and b_i matrices.

    Returns
    -------
    numpy.ndarray
        The M_i matrix.

    """

    M_i = np.block([[A_i, b_i.reshape(-1, 1) / 2], [b_i.reshape(1, -1) / 2, c_i]])

    return M_i


def build_L_i(D, d, i):
    """
    Build the matrix L = [[0, D[i] / 2], [D[i]^T / 2, 0]].

    Parameters
    ----------
    D : numpy.ndarray
        The D matrix.

    Returns
    -------
    numpy.ndarray
        The L matrix.

    """

    L = np.block(
        [
            [np.zeros((D.shape[1], D.shape[1])), D[i].reshape(-1, 1) / 2],
            [D[i].reshape(1, -1) / 2, -d[i]],
        ]
    )

    return L


def build_R_i(i, n):
    """
    Build the matrix UB_i = 1/2 * [[0, ..., 0, 1],[0, ..., 0, 1]].

    Parameters
    ----------
    i : int
        The index of the UB_i matrix.
    n : int
        The size of the UB_i matrix.

    Returns
    -------
    numpy.ndarray
        The UB_i matrix.

    """

    R = np.zeros((n + 1, n + 1))
    R[i, n] = 1 / 2
    R[n, i] = 1 / 2

    return R


def build_B(n):
    """
    Build the ball constraints.

    Parameters
    ----------
    n : int
        The size of the problem.

    Returns
    -------
    list
        The list of ball constraints.

    """

    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        B[i, n] = 1
        B[n, i] = 1

    return B


class DataQCQP:
    def __init__(self, n, m, l, seed=0):
        np.random.seed(seed)
        self.n = n
        self.m = m
        self.l = l

        # Linear constraints
        self.D = build_D(l, n)
        self.d = build_d(self.D, l, n)
        self.L = [build_L_i(self.D, self.d, i) for i in range(l)]

        # Objective
        A_0 = build_A_i(0, n)
        b_0 = build_b_i(0, n)
        c_0 = build_c_i(0)
        self.M_0 = build_M_i(A_0, b_0, c_0)

        print("Size of A_1: ", build_A_i(1, n).shape)
        print("Size of b_1: ", build_b_i(1, n).shape)
        print("Size of c_1: ", build_c_i(1))

        # Quadratic constraints
        self.M = []
        for i in range(1, m + 1):
            A_i = build_A_i(i, n)
            b_i = build_b_i(i, n)
            c_i = build_c_i(i)
            self.M.append(build_M_i(A_i, b_i, c_i))

        # Range constraints
        self.R = [build_R_i(i, n) for i in range(n)]

        # Ball constraints
        self.B = build_B(n)


# Ambrosio et al. data and notation. 

def build_Q(n, density):
    """
    Build a random symmetric matrix with a given density.

    Parameters
    ----------
    n : int
        The size of the matrix.
    density : float
        The density of the matrix.

    Returns
    -------
    numpy.ndarray
        The random symmetric matrix.

    """

    # Initialize the Q matrix with zeros
    Q = np.zeros((n, n))
    # Populate off diagonal upper triangular part of the matrix
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < density:
                Q[i, j] = np.random.uniform(-1/(n * np.sqrt(n)), 1/(n * np.sqrt(n)))

    # Copy the upper triangular part to the lower triangular part leaving the diagonal
    Q = Q + Q.T

    # Make the diagonal -1
    Q = Q - np.eye(n)

    return Q

def build_c(i, n):
    """
    Build the vector c = (c_1, ..., c_n)^T where c_i ∈ (0,1) and is scaled to unit norm.

    Parameters
    ----------
    n : int
        The size of the c vector.

    Returns
    -------
    numpy.ndarray
        The c vector.

    """

    # Seed the random number generator
    np.random.seed(i)

    c = np.random.uniform(0, 1, n)
    c = c / np.linalg.norm(c)

    return c

def build_Q_i(i, n, density):
    """
    Build the matrix Q_i with diiagonal 1/n entries and with density populate upper
    triangular part of the matrix with (-1/n^2, 1/n^2) entries.

    Parameters
    ----------
    i : int
        The index of the Q_i matrix.
    n : int
        The size of the Q_i matrix.
    density : float
        The density of the matrix.

    Returns
    -------
    numpy.ndarray
        The Q_i matrix.

    """

    # Use i for seed
    np.random.seed(i)

    # Initialize the Q matrix with zeros
    Q = np.zeros((n, n))

    # Populate off diagonal upper triangular part of the matrix
    for j in range(n):
        for k in range(j + 1, n):
            if np.random.rand() < density:
                Q[j, k] = np.random.uniform(-1/(n ** 2), 1/(n ** 2))

    # Copy the upper triangular part to the lower triangular part leaving the diagonal
    Q = Q + Q.T

    # Make the diagonal 1/n
    Q = Q + np.eye(n) / n

    return Q

def build_q_i(Q_i, c_i, x):
    """
    Build q_i so that the constraint is satisfied, i.e.

    q_i = x^T Q_i x + c_i^T x

    Parameters
    ----------
    Q_i : numpy.ndarray
        The Q_i matrix.
    c_i : numpy.ndarray
        The c_i vector.
    
    Returns
    -------
    float
        The q_i value.

    """

    q_i = x @ Q_i @ x + c_i @ x

    return q_i + 5
    

class DataQCQP_Ambrosio:
    def __init__(self, n, m, l, density, seed=0):
        np.random.seed(seed)
        self.n = n 
        self.m = m
        # For now no linear constraints
        self.l = 0
        self.D = build_D(l, n)
        self.d = build_d(self.D, l, n)
        self.L = [build_L_i(self.D, self.d, i) for i in range(l)]
        
        # Objective
        A_0 = - build_Q(n, density)
        b_0 = - build_c(0, n)
        self.M_0 = build_M_i(A_0, b_0, 0)

        print("Size of Q_1: ", build_Q_i(1, n, density).shape)
        print("Size of c_1: ", build_c(1, n).shape)
        print("Size of q_1: ", build_q_i(build_Q_i(1, n, density), build_c(1, n), np.random.uniform(0, 1, n)))

        # Choose random x to generate q_i
        x = np.random.uniform(0, 1, n)

        # Quadratic constraints
        self.M = []
        for i in range(1, m + 1):
            Q_i = build_Q_i(i, n, density)
            c_i = build_c(i, n)
            q_i = build_q_i(Q_i, c_i, x)
            self.M.append(build_M_i(Q_i, c_i, q_i))
        
        # Range constraints
        self.R = [build_R_i(i, n) for i in range(n)]
        
        # Ball constraints
        self.B = build_B(n)


def standard_sdp_relaxation(data: DataQCQP, verbose=False):
    """
    Solves the SDP problem of the form

    minimize    <M_0, Z>
    subject to  <M_i, Z> <= 0 for i=1,...,m

                <L_i, Z> - d[i] <= 0 for i=1,...,l

                <R_i, Z> - 1 <=0 for i=0,...,n
                - <R_i, Z> <= 0 for i=0,...,n

                or

                <B, Z> <= 1


                Z => 0
                Z = [[X, x], [x^T, 1]]


    where M_i = [[A_i, b_i],[b_i^T, c_i]], L = [[D, 0], [0, 0]], and
    R_i = 1/2 * [[0, ..., 0, 1],[0, ..., 0, 1]].

    Parameters
    ----------
    M_0 : numpy.ndarray
        The M_0 matrix for objective function.
    M : list
        A list of M_i matrices.
    R : list
        A list of R_i matrices.
    B : numpy.ndarray
        The B matrix for the ball constraints.
    verbose : bool, optional
        If True, the solver will print the log. The default is False.

    Returns
    -------
    solution : dict
        A dictionary with the solution of the SDP problem. The keys are:
            - "X": The solution of the SDP problem.
            - "objective": The value of the objective function.
            - "computation_time": The time to solve the problem.

    """

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = data.M_0[0].shape[0]
        Z = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # Objective:
        # <M_0, Z>
        M.objective(mf.ObjectiveSense.Minimize, mf.Expr.dot(data.M_0, Z))

        # Quadratic constraints:
        # <M_i, Z> <= 0
        quadratic_constraints = []
        for i in range(data.m):
            quadratic_constraints.append(
                M.constraint(mf.Expr.dot(data.M[i], Z), mf.Domain.lessThan(0))
            )

        # Linear constraints:
        # <L_i, Z> <= 0 for i=1,...,l
        linear_constraints = []
        for i in range(data.l):
            linear_constraints.append(
                M.constraint(mf.Expr.dot(data.L[i], Z), mf.Domain.lessThan(0))
            )

        # # Ball constraint
        # # <B, Z> <= 1
        # ball_constraint = M.constraint(mf.Expr.dot(data.B, Z), mf.Domain.lessThan(1))

        # # Range constraints for x
        # # <R_i, Z> - 1 <=0 for i=0,...,n
        # # - <R_i, Z> <= 0 for  i=0,...,n
        # range_constraints = []
        # for i in range(len(data.R)):
        #     range_constraints.append(
        #         M.constraint(
        #             mf.Expr.sub(mf.Expr.dot(data.R[i], Z), 1), mf.Domain.lessThan(0)
        #         )
        #     )
        #     range_constraints.append(
        #         M.constraint(
        #             mf.Expr.sub(0, mf.Expr.dot(data.R[i], Z)), mf.Domain.lessThan(0)
        #         )
        #     )

        # Add constraint for constant term of X.
        constant_matrix = np.zeros((size_psd_variable, size_psd_variable))
        constant_matrix[size_psd_variable - 1, size_psd_variable - 1] = 1
        constant_constraint = M.constraint(
            mf.Expr.dot(constant_matrix, Z), mf.Domain.equalsTo(1)
        )

        if verbose:
            # Increase verbosity
            M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        Z_sol = Z.level()
        Z_sol = Z_sol.reshape(size_psd_variable, size_psd_variable)
        computation_time = end_time - start_time

        solution = {
            "Z": Z_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
        }

        # # Print number of constraints
        # print("Number of quadratic constraints: ", len(quadratic_constraints))
        # print("Number of linear constraints: ", len(linear_constraints))
        # print("Number of range constraints: ", len(range_constraints))
        # print("Number of constant constraint: ", 1)


        # # Print dual solutions
        # print("Duals for quadratic constraints:")
        # for i in range(data.m):
        #     print(quadratic_constraints[i].dual())

        # print("Duals for linear constraints:")
        # for i in range(data.l):
        #     print(linear_constraints[i].dual())

        # # print("Duals for ball constraint:")
        # # print(ball_constraint.dual())
            
        # print("Duals for range constraints:")
        # for i in range(len(range_constraints)):
        #     print(range_constraints[i].dual())

        # print("Duals for constant constraint:")
        # print(constant_constraint.dual())

        # print("Frobeinus norm of the solution: ", np.linalg.norm(Z_sol, "fro"))
        # print("Nuclear norm of the solution: ", np.linalg.norm(Z_sol, 1))

        # print("Nuclear norm of the quadratic matrices:")
        # for i in range(data.m):
        #     print("Nuclear norm of M_{}: {}".format(i, np.linalg.norm(data.M[i], 1)))
        # print("Frobenius norm of the quadratic matrices:")
        # for i in range(data.m):
        #     print("Frobenius norm of M_{}: {}".format(i, np.linalg.norm(data.M[i], "fro")))
                  
        # print("Nuclear norm of the linear matrices:")
        # for i in range(data.l):
        #     print("Nuclear norm of L_{}: {}".format(i, np.linalg.norm(data.L[i], 1)))
        # print("Frobenius norm of the linear matrices:")
        # for i in range(data.l):
        #     print("Frobenius norm of L_{}: {}".format(i, np.linalg.norm(data.L[i], "fro")))

        # # print("Nuclear norm of the ball matrix:")
        # # print("Norm of B: {}".format(np.linalg.norm(data.B, 1)))

        # print("Nuclear norm of the range matrices:")
        # for i in range(len(data.R)):
        #     print("Nuclear norm of R_{}: {}".format(i, np.linalg.norm(data.R[i], 1)))
        # print("Frobenius norm of the range matrices:")
        # for i in range(len(data.R)):
        #     print("Frobenius norm of R_{}: {}".format(i, np.linalg.norm(data.R[i], "fro")))

        # print("Nuclear norm of the constant matrix:")
        # print("Nuclear of constant matrix: {}".format(np.linalg.norm(constant_matrix, 1)))
        # print("Frobenius norm of the constant matrix:")
        # print("Frobenius of constant matrix: {}".format(np.linalg.norm(constant_matrix, "fro")))


        return solution


def random_projection_sdp(data: DataQCQP, projector, slack=True):
    """
    Solves the SDP problem of the form

    minimize    <PM_0P^T, Y> - UBe * sum(ubve) + LBe * sum(lbve) - UBi * sum(ubvi)
    subject to  <PM_iP^T, Y> - ubve[i] <= 0 for i=1,...,m  Quadratic
                <PL_iP^T, Y> - ubve[i] <= 0 for i=1,...,l  Linear

                <PR_iP^T, Y> - 1 - ubve[i] <=0 for i=0,...,n
                - <PRP^T_i, Y> - ubve[i] <= 0 for i=0,...,n

                or

                <PBP^T, Y> - ubve[i] <= 1

                <PCP^T, Y> + lbv[i] - ubve[i] = 1

                Y => 0


    where M_i = [[A_i, b_i],[b_i^T, c_i]], L = [[D, 0], [0, 0]], and
    R_i = 1/2 * [[0, ..., 0, 1],[0, ..., 0, 1]], and P is the random

    Parameters
    ----------
    data : DataQCQP
        The data of the problem.

    Returns
    -------
    solution : dict
        A dictionary with the solution of the SDP problem. The keys are:
            - "Y": The solution of the SDP problem.
            - "objective": The value of the objective function.
            - "computation_time": The time to solve the problem.

    """

    size_psd_variable = "soon"

    # Project the matrices
    M_0 = projector.apply_rp_map(data.M_0)
    M_i = [projector.apply_rp_map(M) for M in data.M]
    L = [projector.apply_rp_map(L) for L in data.L]
    R = [projector.apply_rp_map(R) for R in data.R]
    B = projector.apply_rp_map(data.B)

    with mf.Model("SDP") as M:
        # PSD variable X
        size_psd_variable = M_0[0].shape[0]
        Y = M.variable(mf.Domain.inPSDCone(size_psd_variable))

        # if slack:
        #     # no_constraints = data.m + data.l + 2 * data.n + 1
        #     # # Slack variables
        #     # ubv = M.variable(no_constraints, mf.Domain.greaterThan(0))
        #     # lbv = M.variable(no_constraints, mf.Domain.greaterThan(0))
        #     # # Lower and upper bounds of the dual variables
        #     # epsilon = 0.00001
        #     # dual_lower_bound = -1000000000 - epsilon
        #     # dual_upper_bound = 10000000000 + epsilon

        #     # difference_objective = mf.Expr.sub(
        #     #     mf.Expr.mul(
        #     #         dual_upper_bound, mf.Expr.dot(ubv, np.ones(no_constraints))
        #     #     ),
        #     #     mf.Expr.mul(
        #     #         dual_lower_bound, mf.Expr.dot(lbv, np.ones(no_constraints))
        #     #     ),
        #     # )

        #     no_inequalities = data.m + data.l + 2 * data.n
        #     no_equalities = 1

        #     # Slack variables
        #     ubv = M.variable(no_inequalities + 1, mf.Domain.greaterThan(0))
        #     lbv = M.variable(1, mf.Domain.greaterThan(0))

        #     # Lower and upper bounds of the dual variables
        #     epsilon = 0.00001
        #     dual_lower_bound = -10000000 - epsilon
        #     dual_upper_bound = 10000000 + epsilon

        #     difference_objective = mf.Expr.sub(
        #         mf.Expr.mul(dual_lower_bound, lbv),
        #         mf.Expr.mul(
        #             dual_upper_bound, mf.Expr.dot(ubv, np.ones(no_inequalities + 1))
        #         ),
        #     )

        # Objective:
        # <M_0, Z>
        # if slack:
        #     M.objective(
        #         mf.ObjectiveSense.Minimize,
        #         mf.Expr.add(
        #             mf.Expr.dot(M_0, Y),
        #             difference_objective,
        #         ),
        #     )
        # else:
            # M.objective(mf.ObjectiveSense.Minimize, mf.Expr.dot(projector.apply_rp_map(M_0), Y))
        
        M.objective(mf.ObjectiveSense.Minimize, mf.Expr.dot(M_0, Y))

        # Quadratic constraints:
        # <M_i, Z> <= 0
        for i in range(data.m):
            difference_constraint = 0
            # if slack:
            #     # difference_constraint = mf.Expr.sub(ubv.index(i), lbv.index(i))
            #     difference_constraint = ubv.index(i)
            M.constraint(
                mf.Expr.add(mf.Expr.dot(M_i[i], Y), difference_constraint),
                mf.Domain.lessThan(0),
            )

        # Linear constraints:
        # <L_i, Z> <= 0 for i=1,...,l
        for i in range(data.l):
            # if slack:
            #     # difference_constraint = mf.Expr.sub(
            #     #     ubv.index(data.m + i), lbv.index(data.m + i)
            #     # )
            #     difference_constraint = ubv.index(data.m + i)
            
            M.constraint(
                mf.Expr.add(mf.Expr.dot(L[i], Y), difference_constraint),
                mf.Domain.lessThan(0),
            )

        # # # # Ball constraints
        # M.constraint(mf.Expr.dot(B, Y), mf.Domain.lessThan(1))

        # # Range constraints for x
        # # <R_i, Z> - 1 <=0 for i=0,...,n
        # # - <R_i, Z> <= 0 for  i=0,...,n
        # for i in range(len(R)):
        #     if slack:
        #         # difference_constraint = mf.Expr.sub(
        #         #     ubv.index(data.m + data.l + i), lbv.index(data.m + data.l + i)
        #         # )
        #         difference_constraint = ubv.index(data.m + data.l + i)
           
        #     M.constraint(
        #         mf.Expr.add(
        #             mf.Expr.sub(mf.Expr.dot(R[i], Y), 1), difference_constraint
        #         ),
        #         mf.Domain.lessThan(0),
        #     )
            
        #     if slack:
        #         # difference_constraint = mf.Expr.sub(
        #         #     ubv.index(data.m + data.l + data.n + i),
        #         #     lbv.index(data.m + data.l + data.n + i),
        #         # )
        #         difference_constraint = ubv.index(data.m + data.l + data.n + i)
            
        #     M.constraint(
        #         mf.Expr.add(
        #             mf.Expr.sub(0, mf.Expr.dot(R[i], Y)), difference_constraint
        #         ),
        #         mf.Domain.lessThan(0),
        #     )

        # # Add constraint for constant term of X.
        # if slack:
        #     difference_constraint = mf.Expr.sub(
        #         lbv, ubv.index(data.m + data.l + data.n * 2)
            # )
        constant_matrix = np.zeros((size_psd_variable, size_psd_variable))
        constant_matrix[size_psd_variable - 1, size_psd_variable - 1] = 1
        M.constraint(
            mf.Expr.add(mf.Expr.dot(constant_matrix, Y), difference_constraint),
            mf.Domain.equalsTo(1),
        )

        # M.setLogHandler(sys.stdout)

        start_time = time.time()
        # Solve the problem
        M.solve()
        end_time = time.time()

        # Get the solution
        Y_sol = Y.level()
        Y_sol = Y_sol.reshape(size_psd_variable, size_psd_variable)
        computation_time = end_time - start_time

        solution = {
            "Y": Y_sol,
            "objective": M.primalObjValue(),
            "computation_time": computation_time,
            "size_psd_variable": size_psd_variable,
        }

        return solution


def single_problem_results(data, type="sparse", range=(0.1, 0.5), iterations=5):
    """
    Get the results for a single graph.

    Parameters
    ----------
    graph : Graph
        Graph object.
    type : str
        Type of random projector.

    """

    # Solve unprojected problem
    # ----------------------------------------
    print("\n" + "-" * 80)
    print(
        "Results for a QCQP with {} variables, {} quadratic constraints, and {} linear constraints".format(
            data.n, data.m, data.l
        ).center(
            80
        )
    )
    print("-" * 80)
    print("\n{: <18} {: >10} {: >8} {: >8}".format("Type", "Size X", "Value", "Time"))
    print("-" * 80)

    sdp_solution = standard_sdp_relaxation(data)
    print(
        "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
            "SDP Relaxation",
            sdp_solution["size_psd_variable"],
            sdp_solution["objective"],
            sdp_solution["computation_time"],
        )
    )

    matrix_size = sdp_solution["size_psd_variable"]

    # # Solve identity projector
    # # ----------------------------------------
    # id_random_projector = rp.RandomProjector(matrix_size, matrix_size, type="identity")
    # id_rp_solution = random_projection_sdp(data, id_random_projector, slack=False)
    # print(
    #     "{: <18} {: >10} {: >8.2f} {: >8.2f}".format(
    #         "Identity",
    #         id_rp_solution["size_psd_variable"],
    #         id_rp_solution["objective"],
    #         id_rp_solution["computation_time"],
    #     )
    # )

    # Solve random projectors
    # ----------------------------------------
    for rate in np.linspace(range[0], range[1], iterations):
        slack = False
        random_projector = rp.RandomProjector(
            round(matrix_size * rate), matrix_size, type=type
        )
        rp_solution = random_projection_sdp(data, random_projector, slack=slack)

        print(
            "{: <18.2f} {: >10} {: >8.2f} {: >8.2f}".format(
                rate,
                rp_solution["size_psd_variable"],
                rp_solution["objective"],
                rp_solution["computation_time"],
            )
        )


    print()


if __name__ == "__main__":
    # data = DataQCQP(200, 1, 1, seed=0)
    data = DataQCQP_Ambrosio(1000, 10, 0, 0.9, seed=0)
    # matrix_size = data.M_0[0].shape[0]
    # solution = standard_sdp_relaxation(data, verbose=False)
    # projector = rp.RandomProjector(10, matrix_size, type="sparse")
    # solution_rp = random_projection_sdp(data, projector)
    single_problem_results(data, type="sparse", range=(0.1, 0.5), iterations=5)
