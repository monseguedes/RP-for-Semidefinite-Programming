"""
The module contains functions to generate random projectors.

@author: Monse Guedes Ayala
"""

import numpy as np


class RandomProjector:
    def __init__(self, k, m, type="sparse"):
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
        elif self.type == "identity":
            projector = np.eye(self.m)
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
        projector : numpy.ndarray
            Random projector.

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
