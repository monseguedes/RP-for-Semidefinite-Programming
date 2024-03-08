"""
The module contains functions to generate random projectors.

@author: Monse Guedes Ayala
"""

import numpy as np


class RandomProjector:
    def __init__(self, k, m, type="sparse", seed=0):
        """
        Initializes a random projector of dimension k.

        Parameters
        ----------
        k : int
            Dimension of the projector.
        type : string
            Type of random projector. The default is 'sparse'.

        Returns
        -------
        projector : numpy.ndarray
            Random projector of dimension k.

        Examples
        --------
        >>> RandomProjector(3, 'gaussian')
        array([[ 0.4269, -0.5353, -0.7308],
               [-0.904 ,  0.4029, -0.1421],
               [ 0.0084,  0.7423, -0.6709]])

        >>> RandomProjector(3, 'sparse')
        array([[ 0,  0,  0],
               [ 0,  1,  0],
               [ 0,  0,  0]])

        """

        self.k = k
        self.m = m
        self.type = type
        np.random.seed(seed)
        self.projector = self.generate_random_projector()

    def generate_random_projector(self):
        """
        Generates a squared random projector of a given dimension k.

        Possible types of random projectors:
            - 'gaussian': Gaussian random projector.
            - 'sparse': Sparse random projector, 1, -1, or 0 with
            probability 1/6, 1/6, 2/3, respectively.

        Parameters
        ----------
        dimension : int
            Dimension of the projector.
        type : string
            Type of random projector. The default is 'sparse'.

        Returns
        -------
        projector : numpy.ndarray
            Random projector of dimension k.

        Examples
        --------
        >>> generate_random_projector(3, 'gaussian')
        array([[ 0.4269, -0.5353, -0.7308],
            [-0.904 ,  0.4029, -0.1421],
            [ 0.0084,  0.7423, -0.6709]])

        >>> generate_random_projector(3, 'sparse')
        array([[ 0,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  0]])

        """

        if self.type == "gaussian":
            projector = np.random.randn(self.k, self.m)
        elif self.type == "sparse":
            projector = np.random.choice(
                [-1, 0, 1], size=(self.k, self.m), p=[1 / 6, 2 / 3, 1 / 6]
            )
        elif self.type == "sparser":
            projector = np.random.choice(
                [-1, 0, 1], size=(self.k, self.m), p=[5 / 100, 90 / 100, 5 / 100]
            )

        elif self.type == "identity":
            projector = np.eye(self.m)
        elif self.type == "debug":
            projector = np.random.randint(low=-10, high=10, size=(self.k, self.m))
        elif self.type == "debug_not_0_mean":
            projector = np.random.randint(low=-10, high=10, size=(self.k, self.m))
            projector = projector - np.mean(projector)
        elif self.type == "debug_random_rows":
            projector = np.random.randint(low=-10, high=10, size=(self.k, self.m))
            for i in range(self.k):
                seed = np.random.seed(i)
                lower = np.random.randint(low=-20, high=-1)
                upper = np.random.randint(low=1, high=20)
                projector[i, :] = np.random.randint(
                    low=lower, high=upper, size=(1, self.m)
                )
        elif self.type == "debug_random_columns":
            projector = np.random.randint(low=-10, high=10, size=(self.k, self.m))
            for i in range(self.m):
                seed = np.random.seed(i)
                lower = np.random.randint(low=-20, high=-1)
                upper = np.random.randint(low=1, high=20)
                projector[:, i] = np.random.randint(
                    low=lower, high=upper, size=(1, self.k)
                ) - np.mean(projector[:, i - 1])

        elif self.type == "0.2_density":
            p = 0.2
            standard_deviation = 1 / (np.sqrt(self.k) * p)
            projector = np.random.normal(0, standard_deviation, (self.k, self.m))
            # Make 1 - p entries of the matrix be 0
            random_indices = np.random.choice(projector.size, round(projector.size * (1 - p)), replace=False)
            # Convert the 1D random indices to the corresponding 2D indices
            indices_2d = np.unravel_index(random_indices, projector.shape)
            projector[indices_2d] = 0
            
        elif self.type == "0.1_density":
            p = 0.1
            standard_deviation = 1 / (np.sqrt(self.k) * p)
            projector = np.random.normal(0, standard_deviation, (self.k, self.m))
            # Make 1 - p entries of the matrix be 0
            random_indices = np.random.choice(projector.size, round(projector.size * (1 - p)), replace=False)
            # Convert the 1D random indices to the corresponding 2D indices
            indices_2d = np.unravel_index(random_indices, projector.shape)
            projector[indices_2d] = 0

        elif self.type == "debug_ones":
            projector = np.ones((self.k, self.m))
        elif self.type == "debug_zeros":
            projector = np.zeros((self.k, self.m))
        elif self.type == "debug_not_full_ones":
            projector = np.zeros((self.k, self.m))
            for i in range(self.k):
                projector[i, i] = 1
        elif self.type == "debug_not_full_random":
            projector = np.random.randint(low=-10, high=10, size=(self.k, self.m))
            for i in range(round(self.k * 0.3)):
                projector[i, :] = 0
        else:
            raise ValueError("The type of random projector is not valid.")
        return projector

    def apply_rp_map(self, matrix):
        """
        Applies the random projection map m_P(A):A --> PAP^T to a matrix A.

        Parameters
        ----------
        matrix : numpy.ndarray
            Squared matrix to be projected.

        Returns
        -------
        projected_matrix : numpy.ndarray
            Projected matrix.

        Examples
        --------
        >>> apply_rp_map(np.array([[1, 2], [3, 4]]), np.array([[0, 0], [0, 1]]))
        array([[0, 0],
            [0, 4]])

        """

        return (self.projector @ matrix) @ self.projector.T

    def lift_solution(self, solution):
        """
        Lifts the solution of the projected problem to the original space.

        Parameters
        ----------
        solution : numpy.ndarray
            Solution of the projected problem.

        Returns
        -------
        lifted_solution : numpy.ndarray
            Lifted solution to the original space.

        Examples
        --------
        >>> lift_solution(np.array([[1, 2], [3, 4]]), np.array([[0, 0], [0, 1]]))
        array([[0, 0],
            [0, 4]])

        """

        return self.projector.T @ solution @ self.projector
    
    def make_random_squared_matrix(self, type="sparse"):
        """
        Generates a random squared matrix of a given dimension k.

        Returns
        -------
        matrix : numpy.ndarray
            Random squared matrix of dimension k.

        """

        if type == "gaussian":
            matrix = np.random.randn(self.k, self.k)
        elif type == "sparse":
            matrix = np.random.choice(
                [-1, 0, 1], size=(self.k, self.k), p=[1 / 6, 2 / 3, 1 / 6]
            )
        elif type == "identity":
            matrix = np.eye(self.k)
        elif type == "debug":
            matrix = np.random.randint(low=-10, high=10, size=(self.k, self.k))
        elif type == "debug_constant":
            matrix = np.ones((self.k, self.k))
        elif type == "debug_zeros":
            matrix = np.zeros((self.k, self.k))
        else:
            raise ValueError("The type of random projector is not valid.")
        return matrix
