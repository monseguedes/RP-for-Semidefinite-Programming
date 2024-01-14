"""
The module contains functions to generate random projectors.

@author: Monse Guedes Ayala
"""

import numpy as np


def generate_random_projector(k, type="sparse"):
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

    if type == "gaussian":
        projector = np.random.randn(k, k)
    elif type == "sparse":
        projector = np.random.choice([-1, 0, 1], size=(k, k), p=[1 / 6, 2 / 3, 1 / 6])
    else:
        raise ValueError("The type of random projector is not valid.")
    return projector
