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
"""

import numpy as np
import math

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
    Q_1 = np.eye(n) - 2 * np.outer(w_1, w_1) / np.linalg.norm(w_1)**2
    w_2 = np.random.uniform(-1, 1, n)
    Q_2 = np.eye(n) - 2 * np.outer(w_2, w_2) / np.linalg.norm(w_2)**2
    w_3 = np.random.uniform(-1, 1, n)
    Q_3 = np.eye(n) - 2 * np.outer(w_3, w_3) / np.linalg.norm(w_3)**2

    P_i = Q_1 @ Q_2 @ Q_3

    if i == 0:
        T_i = np.zeros((n, n))
        for i in range(math.floor(n / 2)):
            T_i[i, i] = np.random.uniform(-5 * n, 0)
        for i in range(math.floor(n / 2) + 1, n):
            T_i[i, i] = np.random.uniform(0, 5 * n)

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
        b_i = np.random.uniform(-100, 100, n)

    b_i = np.random.uniform(-50 * n, 0, n)

    return b_i

def build_c_i(m):
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

    c_i = np.random.uniform(-50, 0, m)

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

    D = np.random.uniform(0, 50, (l, n))

    return D

def build_d(l, n):
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

    d = np.ones(l) * l * np.random.uniform(0, 50) / n

    return d

def build_Mi(i, n):
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

    A_i = build_A_i(i, n)
    b_i = build_b_i(i, n)
    c_i = build_c_i(1)

    M_i = np.block([[A_i, b_i.reshape(-1, 1)], [b_i.reshape(1, -1), c_i]])

    return M_i

def build_L(D):
    raise NotImplementedError

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
    R[i, n ] = 1/2
    R[n, i] = 1/2

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
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.M_0 = build_Mi(0, n)
        self.M = [build_Mi(i, n) for i in range(1, m)]
        self.R = [build_R_i(i, n) for i in range(n)]
        self.B = build_B(n)
    

if __name__ == "__main__":
    n = 5
    m = 3
    l = 2

    A_i = build_A_i(0, n)
    b_i = build_b_i(0, n)
    c_i = build_c_i(1)
    D = build_D(l, n)
    d = build_d(l, n)
    M_i = build_Mi(0, n)
    B_i = build_B(0, n)

    print(A_i)
    print(b_i)
    print(c_i)
    print(D)
    print(d)
    print(M_i)
    print(B_i)

    print("Done.")